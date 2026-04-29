"""
train.py — TIR SR training with stratified random split.
- Loads metadata, computes stats, creates datasets and dataloaders.
- Builds model from config, sets up W&B logging and callbacks.
- Runs training and test evaluation, logging results to W&B.
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
        from scripts.models.real_esrgan import RealESRGANModule
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


# ---------------------- Split ---------------------
def stratified_random_split(metadata: pd.DataFrame):
    """70/15/15 split stratified by campaign_id."""
    train_val, test_df = train_test_split(
        metadata, test_size=0.15, random_state=42,
        stratify=metadata["campaign_id"]
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.15/0.85, random_state=42,
        stratify=train_val["campaign_id"]
    )
    print(f"\n  Stratified split:")
    print(f"  Train: {len(train_df)} patches")
    print(f"  Val:   {len(val_df)} patches")
    print(f"  Test:  {len(test_df)} patches")
    print(f"\n  Per-campaign distribution:")
    for cid in sorted(metadata["campaign_id"].unique()):
        n_train = (train_df["campaign_id"] == cid).sum()
        n_val   = (val_df["campaign_id"]   == cid).sum()
        n_test  = (test_df["campaign_id"]  == cid).sum()
        print(f"    {cid:15s} train:{n_train:4d} val:{n_val:4d} test:{n_test:4d}")
    return train_df, val_df, test_df


def save_split(train_df, val_df, test_df, metadata_json: str):
    """Write split assignments back to metadata JSON."""
    with open(metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))

    metadata["split"] = None
    metadata.loc[metadata["patch_name"].isin(train_df["patch_name"]), "split"] = "train"
    metadata.loc[metadata["patch_name"].isin(val_df["patch_name"]),   "split"] = "val"
    metadata.loc[metadata["patch_name"].isin(test_df["patch_name"]),  "split"] = "test"

    with open(metadata_json, "w") as f:
        json.dump(json.loads(metadata.to_json(orient="records", indent=2)), f, indent=2)
    print(f"\n  Split saved to {metadata_json}")


# ---------------------- Core training ---------------------
def run(train_df, val_df, test_df, args, model_cfg):
    print(f"\n{'='*50}")
    print(f"MODEL: {args.model.upper()}")
    print(f"{'='*50}")

    mean, std  = compute_mean_std(train_df)
    data_range = compute_data_range(train_df)

    train_patch_size = model_cfg.get("patch_size", 48)
    if "img_size" in model_cfg and args.model in ["swinir", "hat"]:
        train_patch_size = model_cfg["img_size"]

    train_aug = Compose([
        GeoAugment(),
        TIRNoise(std=std, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.2)),
    ])

    use_aux = "aux_path" in train_df.columns and train_df["aux_path"].notna().any()
    print(f"[INFO] use_aux={use_aux}")

    train_ds = SRDataset(train_df, mean=mean, std=std, patch_size=train_patch_size,
                         is_train=True,  transforms=train_aug, use_aux=use_aux)
    val_ds   = SRDataset(val_df,   mean=mean, std=std,
                         is_train=False, transforms=None,      use_aux=use_aux)
    test_ds  = SRDataset(test_df,  mean=mean, std=std,
                         is_train=False, transforms=None,      use_aux=use_aux)

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,               shuffle=False, **loader_kw)

    # Build model
    cls           = get_model_class(args.model)
    explicit_keys = {
        "mean", "std", "learning_rate", "patience", "pretrained_path",
        "pretrained_g_path", "pretrained_d_path",
        "freeze_backbone", "bb_lr_scale", "lambda_grad",
        "model_class", "precision", "data_range",
    }
    model_hparams           = {k: v for k, v in model_cfg.items() if k not in explicit_keys}
    model_hparams["use_aux"] = use_aux

    init_params = inspect.signature(cls.__init__).parameters
    if "data_range" in init_params:
        model_hparams["data_range"] = data_range

    precision = model_cfg.get("precision", "bf16-mixed")
    print(f"[INFO] Using precision: {precision}")

    if args.model == "real_esrgan":
        model = cls(
            mean=mean, std=std,
            learning_rate=args.lr,
            patience=args.patience,
            pretrained_g_path=args.pretrained,
            pretrained_d_path=args.pretrained_d,
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
            pretrained_path=args.pretrained,
            freeze_backbone=args.freeze_backbone,
            bb_lr_scale=args.bb_lr_scale,
            lambda_grad=args.lambda_grad,
            **model_hparams,
        )

    run_name = args.run_name or args.model.upper()
    group    = args.group    or args.model.upper()

    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=group,
        log_model='all',
        config={**model_cfg, **vars(args)},
    )

    callbacks = [
        EarlyStopping(monitor="val_psnr", patience=args.patience, mode="max"),
        ModelCheckpoint(
            monitor="val_psnr", mode="max", save_top_k=1,
            filename=f"{run_name}-{{epoch:03d}}-{{val_psnr:.2f}}",
            dirpath=f"checkpoints/{run_name}",
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)
    results = trainer.test(model, dataloaders=test_loader, ckpt_path="best")[0]
    print(f"\nTest results: {results}")

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

    print(f"Total patches: {len(metadata)}")
    print(f"Campaigns: {sorted(metadata['campaign_id'].unique().tolist())}")

    start = time.time()

    train_df, val_df, test_df = stratified_random_split(metadata)
    save_split(train_df, val_df, test_df, args.metadata_json)
    run(train_df, val_df, test_df, args, model_cfg)

    print(f"\nCompleted in {(time.time() - start) / 60:.2f} min")


# ---------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="TIR SR training")

    p.add_argument("--model", required=True,
                   choices=["edsr", "swinir", "hat", "real_esrgan"])

    # Data
    p.add_argument("--metadata_json", default="data/full_metadata.json")
    p.add_argument("--pretrained",    default=None)
    p.add_argument("--pretrained_d",  default=None,
                   help="Discriminator checkpoint (real_esrgan only)")

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
    p.add_argument("--run_name", default=None)
    p.add_argument("--group",    default=None)

    return p.parse_args()


if __name__ == "__main__":
    main()