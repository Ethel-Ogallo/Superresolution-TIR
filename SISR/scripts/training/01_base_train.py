"""
01_train.py — fine-tuning for TIR Super-Resolution benchmark.
"""

import argparse
import json
import os
import random
import sys

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


# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Superresolution-TIR
SISR_ROOT = PROJECT_ROOT / "SISR"
SCRIPTS_ROOT = SISR_ROOT / "scripts"

# Allow imports from the SISR directory.
sys.path.insert(0, str(SISR_ROOT))

from scripts.utils.dataset import SRDataset
from scripts.utils.data_aug import train_transforms

PATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "patches"
STATS_PATH = PATCHES_DIR / "stats.json"
PRETRAINED = PROJECT_ROOT / "data" / "pretrained"
CONFIGS_DIR = SISR_ROOT / "configs"
CKPT_DIR = SISR_ROOT / "checkpoints" / "phase2.1"

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
    "fastdiffsr": "fastdiffx4.pth",
    "diffbir_bsrnet":  "BSRNet.pth",
    "diffbir_control": "v2.pth",
    "diffbir_sd":      "v2-1_512-ema-pruned.ckpt",
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

    if model_name == "fastdiffsr":
        from scripts.models.fastdiffsr import FastDiffSRModule
        return FastDiffSRModule

    if model_name == "diffbir":
        from scripts.models.diffbir import DiffBIRModule
        return DiffBIRModule

    raise ValueError(f"Unknown model: {model_name}")


def build_model(model_name, cfg, stats, args):
    cls = get_model_class(model_name)

    # FastDiffSR 
    if model_name == "fastdiffsr":
        model = cls(
            stats_path=str(STATS_PATH),
            learning_rate=args.lr,
            checkpoint_path=str(PRETRAINED / PRETRAINED_PATHS["fastdiffsr"]),
        )
        print( "[INFO] FastDiffSR checkpoint:", PRETRAINED / PRETRAINED_PATHS["fastdiffsr"],)
        return model

    # DiffBIR
    if model_name == "diffbir":
        model = cls(
            stats_json=str(STATS_PATH),
            learning_rate=args.lr,
            bsrnet_checkpoint=str(
                PRETRAINED / PRETRAINED_PATHS["diffbir_bsrnet"]
            ),
            controlnet_checkpoint=str(
                PRETRAINED / PRETRAINED_PATHS["diffbir_control"]
            ),
            sd_checkpoint=str(
                PRETRAINED / PRETRAINED_PATHS["diffbir_sd"]
            ),
            sample_steps=args.diffbir_steps,
        )

        print("[INFO] DiffBIR BSRNet:", PRETRAINED / PRETRAINED_PATHS["diffbir_bsrnet"],)
        print("[INFO] DiffBIR ControlNet:",PRETRAINED / PRETRAINED_PATHS["diffbir_control"],)
        print("[INFO] DiffBIR Stable Diffusion:",PRETRAINED / PRETRAINED_PATHS["diffbir_sd"],)

        return model

    # common arguments and configuration
    pretrained_path = str(PRETRAINED / PRETRAINED_PATHS[model_name])
    common = dict(
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        data_min=stats["hr_percentiles"]["p1"],
        learning_rate=args.lr,
    )

    arch_kwargs = {
        k: v for k, v in cfg.items()
        if k not in {"model_class", "precision"}
    }

    model_kwargs = {
        **common,
        **arch_kwargs,
        "pretrained_path": pretrained_path,
    }

    if model_name == "realesrgan":
        model_kwargs["pretrained_d_path"] = str(PRETRAINED / PRETRAINED_D_PATHS["realesrgan"])

    print(f"[DEBUG] pretrained_path: {pretrained_path}")

    return cls(**model_kwargs)


# Main training function 
def run(args):
    set_seed(args.seed)
    print(f"\n{'═'*60}")
    print(f"Model : {args.model.upper()} — Fine-tuning")
    print(f"{'═'*60}")

    # Load stats and config 
    with open(STATS_PATH) as f:
        stats = json.load(f)

    if args.model == "diffbir":
        cfg = {}
        precision = "16-mixed"
    else:
        cfg = load_config(args.model)
        precision = cfg.get("precision", "bf16-mixed")

    # Datasets 
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=train_transforms(),          
        use_water_mask=True,    
    )
    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=None,
        use_water_mask=True,   
    )
    test_ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=None,
        use_water_mask=True,    
    )

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size,shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size,shuffle=False, **loader_kw)

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
        filename  = f"{args.model}_phase2_{{epoch:02d}}_{{val/water_mae:.4f}}",
        monitor   = "val/water_mae",
        mode      = "min",
        save_top_k= 1,
        verbose   = True,
    )

    callbacks = [
        ckpt_callback,
        EarlyStopping(
            monitor  = "val/water_mae",
            mode     = "min",
            patience = args.patience,
            verbose  = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    val_check_interval = 1
    accumulate_steps = 4 if args.model == "fastdiffsr" else (2 if args.model == "hat" else 1)

    trainer = Trainer(
        max_epochs              = args.max_epochs,
        accelerator             = "gpu",
        devices                 = 1,
        precision               = precision,
        logger                  = wandb_logger,
        callbacks               = callbacks,
        log_every_n_steps       = 10,
        accumulate_grad_batches = accumulate_steps,
    )

    # Train 
    trainer.fit(model, train_loader, val_loader)
    print(f"\n[INFO] Best checkpoint: {ckpt_callback.best_model_path}")

    # test
    trainer.test(model, test_loader, ckpt_path=ckpt_callback.best_model_path)

    wandb.finish()

    return ckpt_callback.best_model_path


#  CLI 
def parse_args():
    p = argparse.ArgumentParser(description="fine-tuning")
    p.add_argument("--model",required=True,
                   choices=["edsr", "swinir", "hat", "realesrgan", "fastdiffsr", "diffbir"])
    p.add_argument(
        "--diffbir_steps",
        type=int,
        default=10,
        help="DiffBIR sampling steps for validation/test."
    )
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--max_epochs",  type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--project",     default="TIR_sisr_final")
    p.add_argument("--run_name",    default=None)
    p.add_argument("--group",       default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)