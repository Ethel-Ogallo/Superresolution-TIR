"""
train.py — TIR SR training.

Split strategies (mutually exclusive):
    --loo                          Leave-one-out across all campaigns (recommended)
    --holdout_campaign BCR_2022    Single fixed campaign holdout
    --random_split                 Random 70/15/15 patch-level split

Usage:
    python -m scripts.training.train --model edsr --loo
    python -m scripts.training.train --model edsr --holdout_campaign BCR_2022
    python -m scripts.training.train --model edsr --random_split
    python -m scripts.training.train --model real_esrgan --loo \
        --pretrained weights/RealESRGAN_x4plus.pth \
        --pretrained_d weights/RealESRGAN_x4plus_netD.pth
"""

import argparse
import json
import os
import random
import time
import inspect

import numpy as np
import pandas as pd
import torch
import wandb
import yaml
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from scripts.utils.dataset import (
    SRDataset,
    compute_mean_std,
    compute_data_range,
    Compose,
    GeoAugment,
    TIRNoise,
    BlurAugment,
    ContrastScaling,
    ThermalShift,
)


# ---------------------- Reproducibility ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------- Config ---------------------
def load_model_config(model_name: str) -> dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "configs", f"{model_name}.yaml"),
        os.path.join(repo_root, "scripts", "training", "configs", f"{model_name}.yaml"),
    ]
    for path in candidates:
        if os.path.exists(path):
            with open(path) as f:
                return yaml.safe_load(f)
    raise FileNotFoundError(f"No config found for '{model_name}'. Looked in: {candidates}")


# ---------------------- Model registry ---------------------
def get_model_class(model_name: str):
    if model_name == "edsr":
        from scripts.models.edsr import EDSRModule
        return EDSRModule
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    if model_name == "hat":
        from scripts.models.hat import HATModule
        return HATModule
    if model_name == "real_esrgan":
        from scripts.models.realesrgan import RealESRGANModule
        return RealESRGANModule
    raise ValueError(f"Unknown model '{model_name}'")


# ---------------------- Campaign ID ---------------------
def _derive_campaign_id(row: pd.Series) -> str:
    for key in ("campaign_id", "hr_image_id", "patch_name", "patch_id"):
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            parts = value.split("_")
            return "_".join(parts[:2]) if len(parts) >= 2 else parts[0]
    return "unknown"


# ---------------------- Split strategies ---------------------
def campaign_split(metadata: pd.DataFrame, holdout_campaign: str):
    """Hold out one campaign for test, random 80/20 on the rest for train/val."""
    test_df   = metadata[metadata["campaign_id"] == holdout_campaign]
    remaining = metadata[metadata["campaign_id"] != holdout_campaign]
    train_idx, val_idx = train_test_split(remaining.index, test_size=0.2, random_state=42)
    train_df  = remaining.loc[train_idx]
    val_df    = remaining.loc[val_idx]
    print(f"  Train campaigns : {sorted(train_df['campaign_id'].unique().tolist())}")
    print(f"  Split — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    return train_df, val_df, test_df


def random_split(metadata: pd.DataFrame):
    """Random 70/15/15 patch-level split ignoring campaigns."""
    train_val, test_df = train_test_split(metadata, test_size=0.15, random_state=42)
    train_df, val_df   = train_test_split(train_val, test_size=0.15/0.85, random_state=42)
    print(f"  Random split — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)}")
    return train_df, val_df, test_df


def save_split_to_metadata(train_df, val_df, test_df, metadata_json: str, strategy: str):
    """Write split assignments back to the source metadata JSON."""
    with open(metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))

    metadata["split"] = None
    metadata["split_strategy"] = strategy
    metadata.loc[metadata["patch_name"].isin(train_df["patch_name"]), "split"] = "train"
    metadata.loc[metadata["patch_name"].isin(val_df["patch_name"]),   "split"] = "val"
    metadata.loc[metadata["patch_name"].isin(test_df["patch_name"]),  "split"] = "test"

    with open(metadata_json, "w") as f:
        json.dump(json.loads(metadata.to_json(orient="records", indent=2)), f, indent=2)
    print(f"  Split ({strategy}) saved to {metadata_json}")


# ---------------------- Core training function ---------------------
def run_fold(
    train_df: pd.DataFrame,
    val_df: pd.DataFrame,
    test_df: pd.DataFrame,
    fold_name: str,
    args,
    model_cfg: dict,
) -> dict:
    """Train and evaluate one fold."""
    print(f"\n{'='*50}")
    print(f"FOLD: {fold_name}")
    print(f"{'='*50}")

    mean, std  = compute_mean_std(train_df)
    data_range = compute_data_range(train_df)

    train_aug = Compose([
        GeoAugment(),
        TIRNoise(std=std, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.2)),
    ])

    train_ds = SRDataset(train_df, mean=mean, std=std, is_train=True,  transforms=train_aug)
    val_ds   = SRDataset(val_df,   mean=mean, std=std, is_train=False, transforms=None)
    test_ds  = SRDataset(test_df,  mean=mean, std=std, is_train=False, transforms=None)

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,               shuffle=False, **loader_kw)

    # ── Build model ──────────────────────────────────────────────────── #
    cls           = get_model_class(args.model)
    explicit_keys = {
        "mean", "std", "learning_rate", "patience", "pretrained_path",
        "pretrained_g_path", "pretrained_d_path",
        "freeze_backbone", "bb_lr_scale", "lambda_grad", "model_class",
        "precision", "data_range",
    }
    model_hparams = {k: v for k, v in model_cfg.items() if k not in explicit_keys}
    init_params   = inspect.signature(cls.__init__).parameters

    if "data_range" in init_params:
        model_hparams["data_range"] = data_range

    precision = model_cfg.get("precision", "bf16-mixed")
    print(f"[INFO] Using precision: {precision} for {args.model}")

    # ── RealESRGAN takes two pretrained paths; all other models take one ─ #
    if args.model == "real_esrgan":
        model = cls(
            mean=mean, std=std,
            learning_rate=args.lr,
            patience=args.patience,
            pretrained_g_path=args.pretrained,    # --pretrained  (generator)
            pretrained_d_path=args.pretrained_d,  # --pretrained_d (discriminator)
            freeze_backbone=args.freeze_backbone,
            bb_lr_scale=args.bb_lr_scale,
            lambda_grad=args.lambda_grad,
            **model_hparams,
        )
    else:
        model = cls(
            mean=mean, std=std,
            learning_rate=args.lr,
            patience=args.patience,
            pretrained_path=args.pretrained,      # --pretrained  (single ckpt)
            freeze_backbone=args.freeze_backbone,
            bb_lr_scale=args.bb_lr_scale,
            lambda_grad=args.lambda_grad,
            **model_hparams,
        )

    run_name = f"{args.run_name or args.model.upper()}_{fold_name}"
    group    = args.group or args.model.upper()

    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=group,
        log_model='all',
        config={**model_cfg, **vars(args), "fold": fold_name},
    )

    callbacks = [
        EarlyStopping(monitor="val_psnr", patience=args.patience, mode="max"),
        ModelCheckpoint(
            monitor="val_psnr", mode="max", save_top_k=1,
            filename=f"{run_name}-{{epoch:03d}}-{{val_psnr:.2f}}",
            dirpath=f"/tmp/checkpoints/{fold_name}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")[0]

    wandb.finish()
    return results


# ---------------------- Main ---------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    os.makedirs("results/logs", exist_ok=True)

    model_cfg = load_model_config(args.model)

    with open(args.metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))
    if "campaign_id" not in metadata.columns:
        metadata["campaign_id"] = metadata.apply(_derive_campaign_id, axis=1)

    campaigns = sorted(metadata["campaign_id"].dropna().unique().tolist())
    print(f"Campaigns found: {campaigns}")

    start = time.time()

    if args.loo:
        all_results = {}
        for campaign in campaigns:
            train_df, val_df, test_df = campaign_split(metadata, campaign)
            all_results[campaign] = run_fold(
                train_df, val_df, test_df, campaign, args, model_cfg
            )
        print("\n" + "="*50)
        print(f"LOO RESULTS — {args.model.upper()}")
        print("="*50)
        metrics = list(next(iter(all_results.values())).keys())
        summary = {}
        for metric in metrics:
            values = [all_results[c][metric] for c in campaigns]
            summary[metric] = {"mean": float(np.mean(values)), "std": float(np.std(values))}
            print(f"{metric}: {np.mean(values):.4f} ± {np.std(values):.4f}")
        out_path = f"results/logs/{args.model}_loo_results.json"
        with open(out_path, "w") as f:
            json.dump({"folds": all_results, "summary": summary}, f, indent=2)
        print(f"Results saved to {out_path}")

    elif args.holdout_campaign:
        train_df, val_df, test_df = campaign_split(metadata, args.holdout_campaign)
        save_split_to_metadata(train_df, val_df, test_df, args.metadata_json, "campaign")
        run_fold(train_df, val_df, test_df, args.holdout_campaign, args, model_cfg)

    else:
        train_df, val_df, test_df = random_split(metadata)
        save_split_to_metadata(train_df, val_df, test_df, args.metadata_json, "random")
        run_fold(train_df, val_df, test_df, "random", args, model_cfg)

    print(f"\nCompleted in {(time.time() - start) / 60:.2f} min")


# ---------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="TIR SR training")

    p.add_argument("--model", required=True,
                   choices=["edsr", "swinir", "hat", "real_esrgan"])

    # Data
    p.add_argument("--metadata_json", default="data/full_metadata.json")
    p.add_argument("--pretrained",    default=None,
                   help="Generator (or single-model) pretrained checkpoint path")
    # ── RealESRGAN only — ignored silently for all other models ──────── #
    p.add_argument("--pretrained_d",  default=None,
                   help="Discriminator pretrained checkpoint path (real_esrgan only)")

    # Split strategy — mutually exclusive
    split_group = p.add_mutually_exclusive_group(required=True)
    split_group.add_argument("--loo", action="store_true",
                             help="Leave-one-out CV across all campaigns")
    split_group.add_argument("--holdout_campaign", default=None,
                             help="Single fixed campaign holdout e.g. BCR_2022")
    split_group.add_argument("--random_split", action="store_true",
                             help="Random 70/15/15 patch-level split ignoring campaigns")

    # Training
    p.add_argument("--lr",              type=float, default=1e-4)
    p.add_argument("--max_epochs",      type=int,   default=100)
    p.add_argument("--patience",        type=int,   default=30)
    p.add_argument("--batch_size",      type=int,   default=4)
    p.add_argument("--num_workers",     type=int,   default=2)
    p.add_argument("--seed",            type=int,   default=42)
    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--bb_lr_scale",     type=float, default=1.0)
    p.add_argument("--lambda_grad",     type=float, default=0.5)

    # W&B
    p.add_argument("--project",  default="TIR_sisr")
    p.add_argument("--run_name", default=None,
                   help="Base run name, fold suffix appended automatically")
    p.add_argument("--group",    default=None,
                   help="W&B group name")

    return p.parse_args()


if __name__ == "__main__":
    main()