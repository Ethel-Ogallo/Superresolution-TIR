"""
train.py — TIR SR training with Leave-One-Out or single-fold campaign splitting.

Models are implemented as LightningModules in scripts/models, with a shared 
training/validation/test step structure. See EDSRModule in edsr.py for an example.

Launch via SLURM: sbatch jobs/train.sh --model edsr --loo
"""

import argparse
import json
import os
import random
import time

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


# ---------------------- Splits ---------------------
def _derive_campaign_id(row: pd.Series) -> str:
    for key in ("campaign_id", "hr_image_id", "patch_name", "patch_id"):
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.split("_")[0]
    return "unknown"


def build_splits(metadata: pd.DataFrame, holdout_campaign: str):
    """
    Split metadata for one LOO fold.
    - Test  : holdout_campaign
    - Train/Val : remaining campaigns, 80/20 random patch split
    
    Returns train_df, val_df, test_df
    """
    test_df = metadata[metadata["campaign_id"] == holdout_campaign]
    remaining = metadata[metadata["campaign_id"] != holdout_campaign]

    train_idx, val_idx = train_test_split(
        remaining.index, test_size=0.2, random_state=42
    )
    train_df = remaining.loc[train_idx]
    val_df   = remaining.loc[val_idx]

    print(
        f"  Split — train: {len(train_df)} | val: {len(val_df)} | test: {len(test_df)} patches"
    )
    print(
        f"  Train campaigns: {sorted(train_df['campaign_id'].unique().tolist())}"
    )
    return train_df, val_df, test_df


# ---------------------- Single fold ---------------------
def run_fold(holdout_campaign: str, metadata: pd.DataFrame, args, model_cfg: dict) -> dict:
    print(f"\n{'='*50}")
    print(f"FOLD — holdout: {holdout_campaign}")
    print(f"{'='*50}")

    train_df, val_df, test_df = build_splits(metadata, holdout_campaign)
    mean, std = compute_mean_std(train_df)
    data_range = compute_data_range(train_df)

    train_aug = Compose([
        GeoAugment(),
        ThermalShift(range_c=2.0),
        ContrastScaling(range_alpha=(0.90, 1.10)),
        TIRNoise(std=std, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.5)),
    ])

    train_ds = SRDataset(train_df, mean=mean, std=std, is_train=True,  transforms=train_aug)
    val_ds   = SRDataset(val_df,   mean=mean, std=std, is_train=False, transforms=None)
    test_ds  = SRDataset(test_df,  mean=mean, std=std, is_train=False, transforms=None)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,               shuffle=False, **loader_kw)

    # Build model
    cls = get_model_class(args.model)
    # Remove keys from model_cfg that are already passed explicitly
    explicit_keys = {"mean", "std", "learning_rate", "patience", "pretrained_path", 
                     "freeze_backbone", "bb_lr_scale", "lambda_grad", "model_class"}
    model_hparams = {k: v for k, v in model_cfg.items() if k not in explicit_keys}
    model = cls(
        mean=mean,
        std=std,
        learning_rate=args.lr,
        patience=args.patience,
        pretrained_path=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        bb_lr_scale=args.bb_lr_scale,
        lambda_grad=args.lambda_grad,
        **model_hparams,
    )

    base_name = args.run_name or f"{args.model.upper()}_LOO"
    run_name  = f"{base_name}_{holdout_campaign}"
    group     = args.group or f"{args.model.upper()}_LOO"
    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=group,
        log_model="all",
        config={**model_cfg, **vars(args), "holdout_campaign": holdout_campaign},
    )

    callbacks = [
        EarlyStopping(monitor="val_psnr", patience=args.patience, mode="max"),
        ModelCheckpoint(
            monitor="val_psnr", mode="max", save_top_k=1,
            filename=f"{run_name}-{{epoch:03d}}-{{val_psnr:.2f}}",
            dirpath=f"/tmp/checkpoints/{holdout_campaign}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
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
    os.makedirs("results", exist_ok=True)

    model_cfg = load_model_config(args.model)

    # Load metadata once
    with open(args.metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))
    if "campaign_id" not in metadata.columns:
        metadata["campaign_id"] = metadata.apply(_derive_campaign_id, axis=1)

    campaigns = sorted(metadata["campaign_id"].dropna().unique().tolist())
    print(f"Campaigns found: {campaigns}")

    start = time.time()

    if args.loo:
        # --- Leave-One-Out ---
        all_results = {}
        for campaign in campaigns:
            all_results[campaign] = run_fold(campaign, metadata, args, model_cfg)

        # Summarise
        print("\n" + "="*50)
        print(f"LOO RESULTS — {args.model.upper()}")
        print("="*50)
        metrics = list(next(iter(all_results.values())).keys())
        summary = {}
        for metric in metrics:
            values = [all_results[c][metric] for c in campaigns]
            mean_v, std_v = np.mean(values), np.std(values)
            summary[metric] = {"mean": mean_v, "std": std_v}
            print(f"{metric}: {mean_v:.4f} ± {std_v:.4f}")

        out_path = f"results/logs/{args.model}_loo_results.json"
        with open(out_path, "w") as f:
            json.dump({"folds": all_results, "summary": summary}, f, indent=2)
        print(f"\nResults saved to {out_path}")

    else:
        # --- Single fold ---
        if not args.holdout_campaign:
            raise ValueError("Provide --holdout_campaign for single fold, or use --loo")
        run_fold(args.holdout_campaign, metadata, args, model_cfg)

    print(f"\nCompleted in {(time.time() - start) / 60:.2f} min")


# ---------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="TIR SR training")

    # Model
    p.add_argument("--model", required=True,
                   choices=["edsr", "swinir", "hat", "real_esrgan"])

    # Data
    p.add_argument("--metadata_json", default="data/full_metadata.json")
    p.add_argument("--pretrained", default=None)

    # Split strategy
    p.add_argument("--loo", action="store_true",
                   help="Leave-one-out cross-validation across all campaigns")
    p.add_argument("--holdout_campaign", default=None,
                   help="Single holdout campaign (only used without --loo)")

    # Training
    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--max_epochs",     type=int,   default=100)
    p.add_argument("--patience",       type=int,   default=30)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)
    p.add_argument("--freeze_backbone",action="store_true")
    p.add_argument("--bb_lr_scale",    type=float, default=1.0)
    p.add_argument("--lambda_grad",    type=float, default=0.5)

    # W&B
    p.add_argument("--project",   default="TIR_sisr")
    p.add_argument("--run_name",  default=None,
                   help="Base run name — fold suffix appended automatically e.g. EDSR_LOO_PDR")
    p.add_argument("--group",     default=None,
                   help="W&B group name (defaults to MODEL_LOO)")
    p.add_argument("--run_test",  action="store_true")

    return p.parse_args()


if __name__ == "__main__":
    main()