# scripts/training/train.py
import argparse
import json
import random
import numpy as np
import torch
import wandb
import yaml
import time
from datetime import timedelta
from pathlib import Path
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (
    EarlyStopping,
    LearningRateMonitor,
    ModelCheckpoint,
)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader
from scripts.utils.dataset import SRDataset
from scripts.utils.data_aug import train_transforms


# =========================================================
# PATHS
# =========================================================
BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH = PATCHES_DIR / "stats.json"
PRETRAINED = BASE / "data/pretrained"
CONFIGS_DIR = BASE / "configs"
CKPT_DIR = BASE / "checkpoints/dev_v4"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# CONFIG LOADING
# =========================================================
def load_config(model_name):
    path = CONFIGS_DIR / f"{model_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# =========================================================
# MODEL LOADER
# =========================================================
def get_model_class(model_name):
    if model_name == "swinir":
        from scripts.models.swinir_copy import SwinIRModule
        return SwinIRModule
    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# BUILD MODEL — UPDATED
# =========================================================
def build_model(model_name, cfg, stats, args, aux_chans):
    cls = get_model_class(model_name)
    pretrained_path = str(PRETRAINED / "SwinIR_classical_x4.pth")

    skip_keys = {"model_class", "precision", "lulc_channels"}
    arch_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}

    return cls(
        **arch_kwargs,
        pretrained_path=pretrained_path,
        learning_rate=args.lr,
        aux_chans=aux_chans,
        lambda_grad=args.lambda_grad,
        lambda_water=args.lambda_water,
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        data_min=stats["hr_percentiles"]["p1"],
        freeze_backbone=bool(args.freeze_backbone),
        freeze_mode=args.freeze_mode,
        # New parameters for SPADE stability
        spade_lr_scale=cfg.get("spade_lr_scale", 0.3),
        # aux_embed_dim=cfg.get("aux_embed_dim", 64),
    )


# =========================================================
# TRAINING LOOP
# =========================================================
def run(args):
    set_seed(args.seed)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg = load_config(args.model)
    precision = cfg.get("precision", "bf16-mixed")
    USE_AUX = bool(args.use_aux)

    transforms = train_transforms() if USE_AUX else None

    # =====================================================
    # DATASETS
    # =====================================================
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "train" / "AUX") if USE_AUX else None,
        repeat_channels=False,
        transform=transforms,
    )

    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "val" / "AUX") if USE_AUX else None,
        repeat_channels=False,
        transform=None,
    )

    # (test_ds can stay as is)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    # =====================================================
    # AUX CHANNEL DETECTION
    # =====================================================
    sample = train_ds[0]
    aux_chans = None
    for key in ["aux", "aux_lr", "aux_mid"]:
        if key in sample:
            aux_chans = sample[key].shape[0]
            break
    if aux_chans is None:
        aux_chans = 0
    print(f"[INFO] AUX channels detected: {aux_chans}")

    # =====================================================
    # MODEL
    # =====================================================
    model = build_model(
        args.model,
        cfg,
        stats,
        args,
        aux_chans,           # Removed lulc_channels
    )


    # =====================================================
    # LOGGER
    # =====================================================
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or f"swinir_spade_direct_aux{int(USE_AUX)}",
        group=args.group,
        config={**cfg, **vars(args)},
    )


    # =====================================================
    # CHECKPOINTING
    # =====================================================
    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR / args.model,
        filename="best-{epoch:02d}-{val_full_psnr:.4f}",
        monitor="val_full_psnr",
        mode="max",
        save_top_k=1,
    )

    callbacks = [
        ckpt,
        EarlyStopping(
            monitor="val_full_psnr",
            mode="max",
            patience=args.patience,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # =====================================================
    # TRAINER
    # =====================================================
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    # =====================================================
    # TRAIN
    # =====================================================
    start = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"\n[INFO] Training time: {timedelta(seconds=int(time.time() - start))}")
    print("\n[INFO] Best checkpoint:", ckpt.best_model_path)
    wandb.finish()
    return ckpt.best_model_path


# =========================================================
# CLI
# =========================================================
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)

    p.add_argument("--project", default="TIR_SISR")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)

    p.add_argument("--use_aux", type=int, default=1)

    p.add_argument("--lambda_grad", type=float, default=0.0)
    p.add_argument("--lambda_water", type=float, default=0.0)

    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--freeze_mode", type=str, default="none")

    p.add_argument(
        "--input_init",
        type=str,
        default="pretrained_mean",
        choices=["pretrained_mean", "gaussian", "xavier", "he", "partial_preserve"],
    )

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args) 