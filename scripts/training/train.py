# scripts/training/train_realesrgan_aux.py

import argparse
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import yaml
import wandb

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

# ============================================================
# PATHS
# ============================================================
BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
CONFIGS_DIR = BASE / "configs"
STATS_PATH = PATCHES_DIR / "stats.json"
PRETRAINED = BASE / "data/pretrained"
CKPT_DIR = BASE / "checkpoints/dev_gan"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(name):
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def build_model(cfg, stats, args, aux_chans):
    from scripts.models.realesrgan import RealESRGANModule

    skip = {"model_class", "precision"}
    arch = {k: v for k, v in cfg.items() if k not in skip}

    arch.pop("lambda_perceptual", None)
    arch.pop("lambda_adversarial", None)

    return RealESRGANModule(
        **arch,
        pretrained_path=str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
        pretrained_d_path=str(PRETRAINED / "RealESRGAN_discriminator_x4.pth"),
        learning_rate=args.lr,
        aux_chans=aux_chans,
        # Direct clean mappings to match model definition:
        lambda_nw=args.lambda_nw,
        lambda_w=args.lambda_w,
        lambda_g=args.lambda_g,
        lambda_perceptual=args.lambda_perceptual,    
        lambda_adversarial=args.lambda_adversarial,  
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        data_min=stats["hr_percentiles"]["p1"],
    )

def run(args):
    set_seed(args.seed)

    print("\n" + "=" * 60)
    print("RealESRGAN + SPADE Training: Targeted River Heterogeneity Setup")
    print("=" * 60)

    stats = json.load(open(STATS_PATH))
    cfg = load_config("realesrgan")
    precision = cfg.get("precision", "bf16-mixed")

    # --------------------------------------------------------
    # DATASETS: Completely clean and aligned
    # --------------------------------------------------------
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,                
        aux_dir=str(PATCHES_DIR / "train" / "AUX"), 
        transform=train_transforms(),
    )

    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,                
        aux_dir=str(PATCHES_DIR / "val" / "AUX"),
        transform=None,
    )

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    aux_chans = train_ds[0]["aux_lr"].shape[0]
    print(f"[INFO] AUX channels: {aux_chans}")

    # --------------------------------------------------------
    # MODEL BUILD
    # --------------------------------------------------------
    model = build_model(cfg, stats, args, aux_chans)

    # --------------------------------------------------------
    # WANDB INITIALIZATION
    # --------------------------------------------------------
    run_name = args.run_name or f"realesrgan_river_eval"
    master_config = {}
    master_config.update(cfg)
    master_config.update(vars(args))

    logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group,
        config=master_config,  
    )

    # --------------------------------------------------------
    # THESIS ALIGNED CHECKPOINTS: Monitor Water Temperature Error
    # --------------------------------------------------------
    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=run_name + "_{epoch:02d}_{val_water_mae:.4f}",
        monitor="val/water_mae",  # Save models that yield the lowest stream Celsius error
        mode="min",               # Lower error = better
        save_top_k=1,
    )

    callbacks = [
        ckpt,
        EarlyStopping(monitor="val/water_mae", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # --------------------------------------------------------
    # TRAINER EXECUTION
    # --------------------------------------------------------
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        logger=logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"\nTraining time: {time.time() - t0:.1f}s")
    print(f"Best Model Path: {ckpt.best_model_path}")

    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--strategy", default="spade")
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--lambda_nw", type=float, default=1.0, 
                   help="L1 multiplier for land terrain pixels")
    p.add_argument("--lambda_w", type=float, default=1.0, 
                   help="L1 multiplier for water channel pixels")
    p.add_argument("--lambda_g", type=float, default=0.1, 
                   help="Multiplier for Sobel spatial boundary gradients")
    
    p.add_argument("--lambda_adversarial", type=float, default=0.1) 
    p.add_argument("--lambda_perceptual", type=float, default=1.0)

    p.add_argument("--project", default="TIR_SISR")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)