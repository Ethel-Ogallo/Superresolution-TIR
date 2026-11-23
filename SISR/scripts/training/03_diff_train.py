# scripts/training/03_diff_train.py

"""
Train FastDiffSR x4 with multiscale auxiliary data,
SPADE conditioning, and custom reconstruction loss.
"""

import argparse
import json
import random
import time
import sys
from pathlib import Path
import numpy as np
import torch
import wandb

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import (EarlyStopping, LearningRateMonitor, ModelCheckpoint,)
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader


# PATHS
BASE = Path(__file__).resolve().parents[3]
SISR_ROOT = BASE / "SISR"

sys.path.insert(0, str(SISR_ROOT))

from scripts.utils.dataset3 import SRDataset
from scripts.utils.data_aug import train_transforms
from scripts.models.fastdiffsr_spade import FastDiffSRModule


PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH = PATCHES_DIR / "stats.json"
FASTDIFFSR_ROOT = SISR_ROOT / "scripts/external/FastDiffSR"
FASTDIFFSR_CONFIG = (FASTDIFFSR_ROOT / "config/sr_fastdiffsr_test_64_256.json")
FASTDIFFSR_CHECKPOINT = BASE / "data/pretrained/fastdiffx4.pth"
CKPT_DIR = SISR_ROOT / "checkpoints/fastdiff"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# REPRODUCIBILITY
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


# MODEL
def build_model(args):
    return FastDiffSRModule(
        stats_path=str(STATS_PATH),
        learning_rate=args.lr,
        weight_decay=args.weight_decay,
        checkpoint_path=FASTDIFFSR_CHECKPOINT,
        config_path=FASTDIFFSR_CONFIG,
        lambda_nw=args.lambda_nw,
        lambda_w=args.lambda_w,
        lambda_g=args.lambda_g,
        use_spade=args.use_spade,
    )


# TRAINING
def run(args):
    set_seed(args.seed)
    model_name = "fastdiff_spade" if args.use_spade else "fastdiff_guided"
    run_name = args.run_name or f"{model_name}_{int(time.time())}"
    stats = json.load(open(STATS_PATH))

    # Datasets
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

    test_ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "test" / "AUX"),
        transform=None,
    )

    loader_kw = {"num_workers": args.num_workers,"pin_memory": True,}

    train_loader = DataLoader(train_ds, batch_size=args.batch_size,shuffle=True,**loader_kw,)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size,shuffle=False, **loader_kw,)
    test_loader = DataLoader(test_ds,batch_size=args.batch_size, shuffle=False,**loader_kw,)


    # Model
    model = build_model(args)

    # W&B
    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group,
        config={
            **vars(args),
            "model": model_name,
            "use_spade": args.use_spade,
            "aux_channels": 23,
            "temporal_channels": 2,
            "denoiser_input_channels": 31,
        },
    )

    # Checkpointing
    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=(
            run_name
            + "_epoch={epoch:02d}_val_water_mae={val/water_mae:.4f}"
        ),
        monitor="val/water_mae",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
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

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=args.precision,
        accumulate_grad_batches=4,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=10,
    )

    trainer.fit(model,train_loader, val_loader)
    print(f"Best val/water_mae: {ckpt.best_model_score:.4f}")
    
    # test
    trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)

    wandb.finish()

    return ckpt.best_model_path

# CLI
def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-5)
    p.add_argument("--weight_decay", type=float, default=1e-6)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=1)
    p.add_argument("--patience", type=int, default=15)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--precision", type=str, default="32-true")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_nw", type=float, default=1.0)
    p.add_argument("--lambda_w", type=float, default=2.0)
    p.add_argument("--lambda_g", type=float, default=0.1)
    p.add_argument("--use_spade",action=argparse.BooleanOptionalAction,default=True,)
    p.add_argument("--project", default="TIR_SISR_final")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)
    return p.parse_args()

if __name__ == "__main__":
    args = parse_args()
    run(args)