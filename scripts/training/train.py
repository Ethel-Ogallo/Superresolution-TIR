"""
train.py — Unified training script for TIR SR models.

Usage:
    python train.py --model edsr [common args]
    python train.py --model swinir [common args]
    python train.py --model hat [common args]
    python train.py --model real_esrgan [common args]

Model-specific hyperparameters live in configs/{model}.yaml.
Common training hyperparameters (lr, batch_size, etc.) are CLI args.

Launch via SLURM: sbatch jobs/train.sh --model edsr
"""

import argparse
import json
import os
import random
import time
from typing import Optional

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
    Compose,
    GeoAugment,
    TIRNoise,
    BlurAugment,
    ContrastScaling,
    ThermalShift,
)

# ---------------------- Model registry ---------------------
def get_registry(model_name=None):
    registry = {}

    if model_name == "edsr" or model_name is None:
        from scripts.models.edsr import EDSRModule
        registry["edsr"] = EDSRModule

    if model_name == "swinir" or model_name is None:
        from scripts.models.swinir import SwinIRModule
        registry["swinir"] = SwinIRModule

    if model_name == "hat" or model_name is None:
        from scripts.models.hat import HATModule
        registry["hat"] = HATModule

    if model_name == "real_esrgan" or model_name is None:
        from scripts.models.realesrgan import RealESRGANModule
        registry["real_esrgan"] = RealESRGANModule

    return registry


# ---------------------- Config loading ---------------------
def load_model_config(model_name: str) -> dict:
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    candidates = [
        os.path.join(repo_root, "configs", f"{model_name}.yaml"),
        os.path.join(repo_root, "scripts", "training", "configs", f"{model_name}.yaml"),
    ]

    for config_path in candidates:
        if os.path.exists(config_path):
            with open(config_path) as f:
                return yaml.safe_load(f)

    raise FileNotFoundError(
        "No config found for model "
        f"'{model_name}'. Looked in: {', '.join(candidates)}"
    )


# ---------------------- Reproducibility ---------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------- Splits ---------------------
def _derive_campaign_id(row: pd.Series) -> str:
    """Derive campaign ID from metadata fields like BRC_2023 -> BRC."""
    for key in ("campaign_id", "hr_image_id", "patch_name", "patch_id"):
        value = row.get(key, None)
        if isinstance(value, str) and value.strip():
            return value.split("_")[0]
    return "unknown"


def build_splits(
    metadata_json: str,
    split_mode: str = "campaign",
    holdout_campaign: Optional[str] = None,
    val_campaign: Optional[str] = None,
    force_resplit: bool = False,
):
    with open(metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))

    if "campaign_id" not in metadata.columns:
        metadata["campaign_id"] = metadata.apply(_derive_campaign_id, axis=1)

    need_split = (
        force_resplit
        or "split" not in metadata.columns
        or metadata["split"].isna().all()
    )

    if need_split:
        metadata["split"] = None

        if split_mode == "campaign":
            campaigns = sorted(metadata["campaign_id"].dropna().unique().tolist())
            if not campaigns:
                raise ValueError("No campaign IDs found in metadata.")

            if holdout_campaign is None:
                holdout_campaign = campaigns[-1]
                print(f"[INFO] Auto-selected holdout campaign: {holdout_campaign}")
            if holdout_campaign not in campaigns:
                raise ValueError(
                    f"Holdout campaign '{holdout_campaign}' not found. Available: {campaigns}"
                )

            remaining = [c for c in campaigns if c != holdout_campaign]
            if not remaining:
                raise ValueError("Need at least one campaign besides holdout for training.")

            if val_campaign is not None and val_campaign not in remaining:
                raise ValueError(
                    f"Validation campaign '{val_campaign}' must be in training campaigns: {remaining}"
                )

            if val_campaign is None and len(remaining) >= 2:
                val_campaign = remaining[-1]
                print(f"[INFO] Auto-selected validation campaign: {val_campaign}")

            metadata.loc[metadata["campaign_id"] == holdout_campaign, "split"] = "test"

            if val_campaign is not None:
                metadata.loc[metadata["campaign_id"] == val_campaign, "split"] = "val"
                train_campaigns = [c for c in remaining if c != val_campaign]
                metadata.loc[metadata["campaign_id"].isin(train_campaigns), "split"] = "train"
            else:
                # If only one campaign is left for train/val, split it by patches.
                pool_idx = metadata[metadata["campaign_id"].isin(remaining)].index
                train_idx, val_idx = train_test_split(pool_idx, test_size=0.2, random_state=42)
                metadata.loc[train_idx, "split"] = "train"
                metadata.loc[val_idx, "split"] = "val"

            print(
                "[INFO] Campaign split config: "
                f"train={sorted(metadata[metadata['split']=='train']['campaign_id'].unique().tolist())}, "
                f"val={sorted(metadata[metadata['split']=='val']['campaign_id'].unique().tolist())}, "
                f"test={[holdout_campaign]}"
            )
        else:
            train_val, test = train_test_split(metadata, test_size=0.15, random_state=42)
            train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=42)
            metadata.loc[train.index, "split"] = "train"
            metadata.loc[val.index, "split"] = "val"
            metadata.loc[test.index, "split"] = "test"

        metadata.to_json(metadata_json, orient="records", indent=2)
        print(f"[INFO] Splits written back to {metadata_json}")

    train = metadata[metadata["split"] == "train"]
    val = metadata[metadata["split"] == "val"]
    test = metadata[metadata["split"] == "test"]

    print(f"[INFO] Splits — train: {len(train)} | val: {len(val)} | test: {len(test)}")
    return train, val, test


# ---------------------- Model instantiation ---------------------
def build_model(model_name: str, model_cfg: dict, common_cfg: dict, mean: float, std: float):
    """
    Instantiate the correct LightningModule safely merging CLI/common args and YAML model-specific args.
    CLI/common args take precedence over YAML if a key exists in both.
    """
    registry = get_registry(model_name)
    if model_name not in registry:
        raise ValueError(
            f"Unknown model '{model_name}'. Available: {list(registry.keys())}"
        )
    cls = registry[model_name]

    shared = dict(
        mean=mean,
        std=std,
        learning_rate=common_cfg["lr"],
        patience=common_cfg["patience"],
        pretrained_path=common_cfg.get("pretrained"),
        freeze_backbone=common_cfg.get("freeze_backbone", False),
        bb_lr_scale=common_cfg.get("bb_lr_scale", 1.0),
        lambda_grad=common_cfg.get("lambda_grad", 0.0),
    )

    model_hparams = {k: v for k, v in model_cfg.items() if k != "model_class" and k not in shared}
    return cls(**shared, **model_hparams)

# ---------------------- Main ---------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    os.makedirs("results", exist_ok=True)

    if args.pretrained and not os.path.exists(args.pretrained):
        raise FileNotFoundError(f"Pretrained weights not found: {args.pretrained}")

    model_cfg = load_model_config(args.model)
    print(f"[INFO] Loaded config for '{args.model}': {model_cfg}")

    train_df, val_df, test_df = build_splits(
        metadata_json=args.metadata_json,
        split_mode=args.split_mode,
        holdout_campaign=args.holdout_campaign,
        val_campaign=args.val_campaign,
        force_resplit=args.force_resplit,
    )
    mean, std = compute_mean_std(train_df)

    train_aug = Compose([
        GeoAugment(),
        ThermalShift(range_c=2.0),
        ContrastScaling(range_alpha=(0.90, 1.10)),
        TIRNoise(std=std, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.5)),
    ])

    train_ds = SRDataset(train_df, mean=mean, std=std, is_train=True, transforms=train_aug)
    val_ds = SRDataset(val_df, mean=mean, std=std, is_train=False, transforms=None)
    test_ds = SRDataset(test_df, mean=mean, std=std, is_train=False, transforms=None)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, **loader_kw)

    common_cfg = dict(
        lr=args.lr,
        patience=args.patience,
        pretrained=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        bb_lr_scale=args.bb_lr_scale,
        lambda_grad=args.lambda_grad,
    )
    model = build_model(args.model, model_cfg, common_cfg, mean, std)

    run_name = args.run_name or f"{args.model.upper()}_lr{args.lr:.0e}_bs{args.batch_size}"

    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group or args.model.upper(),
        log_model="all",
        config={**model_cfg, **vars(args)},
    )

    callbacks = [
        EarlyStopping(
            monitor="val_psnr",
            patience=args.patience,
            mode="max",
            verbose=True,
        ),
        ModelCheckpoint(
            monitor="val_psnr",
            mode="max",
            save_top_k=1,
            filename=f"{run_name}-{{epoch:03d}}-{{val_psnr:.2f}}",
            dirpath="/tmp/checkpoints",
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

    print(f"[INFO] Starting run: {run_name}")
    start = time.time()
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    if args.run_test:
        print("[INFO] Evaluating on test set ...")
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    print(f"[INFO] Completed in {(time.time() - start) / 60:.2f} min")
    wandb.finish()


# --------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser(description="TIR SR training, select model with --model")

    p.add_argument("--model", required=True, choices=["edsr", "swinir", "hat", "real_esrgan"],
                   help="Model architecture to train")

    p.add_argument("--metadata_json", default="metadata.json")
    p.add_argument("--pretrained", default=None, help="Path to pretrained weights (optional)")
    p.add_argument("--split_mode", choices=["random", "campaign"], default="campaign",
                   help="Split strategy: random patch-level or campaign-level holdout")
    p.add_argument("--holdout_campaign", default=None,
                   help="Campaign ID to hold out for test (e.g. PDR, DZM, BRC)")
    p.add_argument("--val_campaign", default=None,
                   help="Campaign ID to use for validation (optional)")
    p.add_argument("--force_resplit", action="store_true",
                   help="Force recompute and overwrite split labels in metadata")

    p.add_argument("--lr", type=float, default=1e-4, help="Base learning rate for the optimizer")
    p.add_argument("--max_epochs", type=int, default=20, help="Maximum number of training epochs")
    p.add_argument("--patience", type=int, default=10, help="Number of epochs to wait for improvement")
    p.add_argument("--batch_size", type=int, default=1, help="Batch size for training")
    p.add_argument("--num_workers", type=int, default=2, help="Number of DataLoader workers")
    p.add_argument("--seed", type=int, default=42, help="Random seed for reproducibility")
    p.add_argument("--run_test", action="store_true", help="Run test set evaluation after training")
    p.add_argument("--freeze_backbone", action="store_true", help="Freeze backbone weights during training")
    p.add_argument("--bb_lr_scale", type=float, default=1.0, help="Backbone learning rate scaling factor")
    p.add_argument("--lambda_grad", type=float, default=0.0, help="Weight for gradient-based loss")

    p.add_argument("--project", default="TIR_sisr", help="W&B project name")
    p.add_argument("--run_name", default=None, help="W&B run name (defaults to model + key hparams)")
    p.add_argument("--group", default=None, help="W&B group (defaults to model name)")

    return p.parse_args()


if __name__ == "__main__":
    main()

