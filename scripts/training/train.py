# scripts/training/train.py

import argparse
import json
import os
import random

import numpy as np
import torch
import wandb
import yaml

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


# Paths
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"
CONFIGS_DIR = BASE / "configs"
CKPT_DIR    = BASE / "checkpoints/model_dev/phase1"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# Config
def load_config(model_name):
    path = CONFIGS_DIR / f"{model_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# Load model class dynamically
def get_model_class(model_name):
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    raise ValueError(f"Unknown model: {model_name}")


# Build model
def build_model(model_name, cfg, stats, args):

    cls = get_model_class(model_name)

    pretrained_path = str(PRETRAINED / "SwinIR_classical_x4.pth")

    # remove architecture-only keys safely
    skip_keys = {"model_class", "precision"}
    arch_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}

    model_kwargs = {
        **arch_kwargs,
        "pretrained_path": pretrained_path,
        "learning_rate": args.lr,
        "phase": 2,
        "hr_mean": stats["hr"]["mean"],
        "hr_std": stats["hr"]["std"],
        "data_range": stats["hr_data_range"],
        "adaptation_strategy": args.adaptation_strategy,
    }

    return cls(**model_kwargs)


# Train
def run(args):

    set_seed(args.seed)

    print("\n" + "="*60)
    print(f"Model: {args.model} — Projection Experiment")
    print("="*60)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg = load_config(args.model)
    precision = cfg.get("precision", "bf16-mixed")

    USE_AUX = bool(args.use_aux)

    # Transform for train only
    transforms = train_transforms() if USE_AUX else None

    # Datasets + Loaders
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        aux_dir=str(PATCHES_DIR / "train" / "AUX") if USE_AUX else None,
        repeat_channels=False if USE_AUX else True,
        transform=transforms,
    )

    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        aux_dir=str(PATCHES_DIR / "val" / "AUX") if USE_AUX else None,
        repeat_channels=False if USE_AUX else True,
        transform=None,
    )


    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        **loader_kw
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        **loader_kw
    )


    # model
    model = build_model(args.model, cfg, stats, args)


    # wandb logger + callback setup
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or f"{args.model}_aux{int(USE_AUX)}",
        group=args.group,
        config={**cfg, **vars(args)},
    )

    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR / args.model,
        filename="model_dev-{epoch:02d}-{val_full_psnr:.4f}",
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


    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    trainer.fit(model, train_loader, val_loader)

    wandb.finish()

    print("[INFO] Best checkpoint:", ckpt.best_model_path)

    return ckpt.best_model_path


# CLI
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True, choices=["swinir"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="TIR_SISR")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--use_aux", type=int, default=1, help="1 = TIR+AUX projection, 0 = TIR only")
    p.add_argument("--adaptation_strategy",type=str,default="projection",choices=["projection", "direct"],)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)