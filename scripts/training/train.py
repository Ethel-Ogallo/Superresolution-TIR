# scripts/training/train.py
"""
train.py — Phase 2 fine-tuning for TIR Super-Resolution benchmark.

Usage:
    python -m scripts.training.train --model edsr --phase 2
    python -m scripts.training.train --model swinir --phase 2
    python -m scripts.training.train --model hat --phase 2
    python -m scripts.training.train --model realesrgan --phase 2

SLURM:
    sbatch scripts/training/train_phase2.sh
"""

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

# Paths 
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"
CONFIGS_DIR = BASE / "configs"
CKPT_DIR    = BASE / "checkpoints/phase2"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility 
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# config
def load_config(model_name):
    path = CONFIGS_DIR / f"{model_name}.yaml"
    with open(path) as f:
        cfg = yaml.safe_load(f)
    return cfg


#  Model registry 
PRETRAINED_PATHS = {
    "edsr":       "EDSR_baseline_x4.pth",
    "swinir":     "SwinIR_classical_x4.pth",
    "hat":        "HAT_imagenet_x4.pth",
    "realesrgan": "RealESRGAN_generator_x4.pth",
    "resshift":   "ResShift_x4.pth",    
}

PRETRAINED_D_PATHS = {
    "realesrgan": "RealESRGAN_discriminator_x4.pth",
}


def get_model_class(model_name):
    if model_name == "edsr":
        from scripts.models.edsr import EDSRModule
        return EDSRModule
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    if model_name == "hat":
        from scripts.models.hat import HATModule
        return HATModule
    if model_name == "realesrgan":
        from scripts.models.realesrgan import RealESRGANModule
        return RealESRGANModule
    if model_name == "resshift":
        from scripts.models.resshift import ResShiftModule
        return ResShiftModule
    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name, cfg, stats, args):
    cls = get_model_class(model_name)

    # Common kwargs for all models
    common = dict(
        hr_mean       = stats["hr"]["mean"],
        hr_std        = stats["hr"]["std"],
        data_range    = stats["hr_data_range"],
        learning_rate = args.lr,
        phase         = 2,
    )

    # Pretrained paths
    pretrained_path = str(PRETRAINED / PRETRAINED_PATHS[model_name]) \
                      if model_name in PRETRAINED_PATHS else None

    # Architecture hparams from config
    # Strip non-architecture keys
    skip_keys = {"model_class", "precision"}
    arch_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}

    model_kwargs = {**common, **arch_kwargs,
                    "pretrained_path": pretrained_path}

    # RealESRGAN also needs discriminator path
    if model_name == "realesrgan":
        model_kwargs["pretrained_d_path"] = str(
            PRETRAINED / PRETRAINED_D_PATHS["realesrgan"]
        )
    if model_name == "resshift":
        model_kwargs["ae_path"] = str(PRETRAINED / "autoencoder_vq_f4.pth")
    
    print(f"[DEBUG] pretrained_path: {model_kwargs.get('pretrained_path')}")
    print(f"[DEBUG] ae_path: {model_kwargs.get('ae_path')}")

    return cls(**model_kwargs)


# Main training function 
def run(args):
    set_seed(args.seed)

    print(f"\n{'═'*60}")
    print(f"Model : {args.model.upper()} — Phase 2 Fine-tuning")
    print(f"{'═'*60}")

    # Load stats and config 
    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg = load_config(args.model)
    precision = cfg.get("precision", "bf16-mixed")

    # Datasets 
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=None,          # augmentations added in Phase 2 model dev
    )
    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=None,
    )

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        shuffle=True, **loader_kw
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size,
        shuffle=False, **loader_kw
    )

    #  Model 
    model = build_model(args.model, cfg, stats, args)

    #  WandB logger 
    wandb_logger = WandbLogger(
        project = args.project,
        name    = args.run_name or f"{args.model}_phase2",
        group   = args.group   or f"phase2_{args.model}",
        config  = {**cfg, **vars(args)},
    )

    #  Callbacks 
    ckpt_callback = ModelCheckpoint(
        dirpath   = CKPT_DIR / args.model,
        filename  = f"{args.model}_phase2_{{epoch:02d}}_{{val_psnr:.4f}}",
        monitor   = "val_psnr",
        mode      = "max",
        save_top_k= 1,
        verbose   = True,
    )

    callbacks = [
        ckpt_callback,
        EarlyStopping(
            monitor  = "val_psnr",
            mode     = "max",
            patience = args.patience,
            verbose  = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    val_check_interval = 5 if args.model == "resshift" else 1

    trainer = Trainer(
        max_epochs              = args.max_epochs,
        accelerator             = "gpu",
        devices                 = 1,
        precision               = precision,
        logger                  = wandb_logger,
        callbacks               = callbacks,
        log_every_n_steps       = 5,
        check_val_every_n_epoch = val_check_interval,  # add this
        num_sanity_val_steps=0,
    )

    # Train 
    trainer.fit(model, train_loader, val_loader)

    print(f"\n[INFO] Best checkpoint: {ckpt_callback.best_model_path}")
    print(f"[INFO] Best val_psnr:   {ckpt_callback.best_model_score:.4f}")

    wandb.finish()

    return ckpt_callback.best_model_path


#  CLI 
def parse_args():
    p = argparse.ArgumentParser(description="Phase 2 fine-tuning")

    p.add_argument("--model",       required=True,
                   choices=["edsr", "swinir", "hat", "realesrgan", "resshift"])
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--max_epochs",  type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--project",     default="TIR_sisr")
    p.add_argument("--run_name",    default=None)
    p.add_argument("--group",       default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)