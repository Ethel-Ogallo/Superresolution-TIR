# scripts/training/02_diff_train.py

"""
Train FastDiffSR x4 with auxiliary data .
"""

import argparse
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml
import json
import os
import sys

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint,)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader


# Project directories
BASE = Path(__file__).resolve().parents[3]  # Superresolution-TIR
SISR_ROOT = BASE / "SISR"
SCRIPTS_ROOT = SISR_ROOT / "scripts"

# Allow imports from the SISR directory.
sys.path.insert(0, str(SISR_ROOT))

from scripts.utils.dataset2 import SRDataset
from scripts.utils.data_aug import train_transforms

PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH = PATCHES_DIR / "stats.json"
PRETRAINED_DIR = BASE / "data/pretrained"
FASTDIFFSR_ROOT = BASE / "SISR/scripts/external/FastDiffSR"
FASTDIFFSR_CONFIG = (FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json")
FASTDIFFSR_CHECKPOINT = PRETRAINED_DIR / "fastdiffx4.pth"
CKPT_DIR = BASE / "checkpoints/fastdiff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# Reproducibility
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# Model builder
def build_model(args):
    from scripts.models.fastdiffsr import FastDiffSRModule
    return FastDiffSRModule(
        stats_path=str(STATS_PATH),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=str(FASTDIFFSR_CHECKPOINT),
        config_path=str(FASTDIFFSR_CONFIG),
    )


#  --------------- Main training function --------------------
def run(args):
    set_seed(args.seed)
    print(f"\n{'═'*60}")
    print("FastDiffSR with Aux")
    print(f"\n{'═'*60}")

    # Check paths
    for path in [
        STATS_PATH,
        FASTDIFFSR_CONFIG,
        FASTDIFFSR_CHECKPOINT,
    ]:
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

    #  Datasets
    print("[INFO] Loading datasets...")
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "train" / "AUX_SPADE_HR"),
        repeat_channels=False,
        transform=train_transforms(),
    )

    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "val" / "AUX_SPADE_HR"),
        repeat_channels=False,
        transform=None,
    )

    test_ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "test" / "AUX_SPADE_HR"),
        repeat_channels=False,
        transform=None,
    )


    #  Check one dataset sample
    sample = train_ds[0]
    print("\n[INFO] Checking dataset sample...")
    print("LR:", sample["lr"].shape)
    print("HR:", sample["hr"].shape)
    print("HR mask:", sample["hr_mask"].shape)
    print("AUX:", sample["aux"].shape)
    print("Time channels:", sample["time_channels"].shape)

    assert sample["aux"].shape[0] == 23
    assert sample["time_channels"].shape[0] == 2

    assert sample["aux"].shape[-2:] == (256, 256)
    assert sample["time_channels"].shape[-2:] == (256, 256)

    print("[INFO] Dataset sample passed.")

    # --------------------------------------------------------
    # DataLoaders
    # --------------------------------------------------------
    loader_kw = { "num_workers": args.num_workers,"pin_memory": True, }
    train_loader = DataLoader(train_ds,batch_size=args.batch_size,shuffle=True,**loader_kw, )
    val_loader = DataLoader( val_ds,batch_size=args.batch_size,shuffle=False,**loader_kw,)
    test_loader = DataLoader(test_ds,batch_size=args.batch_size,shuffle=False, **loader_kw,)

    # -Model
    model = build_model(args)

    # Run name
    name = args.run_name or "fastdiffsr_aux"

    # W&B logger
    wandb_logger = WandbLogger(
        project=args.project,
        name=name,
        group=args.group or "fastdiffsr_aux",
        config=vars(args),
    )

    # callbacks
    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=f"{name}_{{epoch:02d}}_{{val/water_mae:.4f}}",
        monitor="val/water_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
        verbose=True,
    )

    callbacks = [
        ckpt,
        EarlyStopping(
            monitor  = "val/water_mae",
            mode     = "min",
            patience = args.patience,
            verbose  = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    #  Trainer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU not available.")

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
        num_sanity_val_steps=0,
        limit_val_batches=args.limit_val_batches,
        limit_train_batches=1.0,
    )

    #  Training
    trainer.fit(model,train_loader, val_loader)
    print(f"Best val/water_mae: {ckpt.best_model_score:.4f}")
    
    # test
    trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)

    wandb.finish()

    return ckpt.best_model_path


# CLI

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument( "--lr", type=float, default=1e-5, )
    p.add_argument("--weight_decay", type=float,default=1e-6,)
    p.add_argument("--batch_size", type=int, default=1,)
    p.add_argument("--max_epochs",type=int, default=1,)
    p.add_argument(  "--num_workers", type=int,default=4,)
    p.add_argument( "--seed",type=int, default=42,)
    p.add_argument("--precision",type=str, default="32-true", )
    p.add_argument("--limit_val_batches", type=int, default=2,)
    p.add_argument("--project", default="TIR_SISR_final", )
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None,)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)