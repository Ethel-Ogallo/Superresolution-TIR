# scripts/training/train_vsr.py

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
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from torch.utils.data import DataLoader

from scripts.utils.seq_dataset import SRDataset
from scripts.models.basicvsrplus import BasicVSRModule

# ============================================================
# PATHS
# ============================================================
BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/seq_patches"
CONFIGS_DIR = BASE / "configs"
STATS_PATH = BASE / "data/processed/patches/stats.json"
PRETRAINED = BASE / "data/pretrained"
CKPT_DIR = BASE / "checkpoints/vsr"
CKPT_DIR.mkdir(parents=True, exist_ok=True)

REQUIRED_STATS_KEYS = [
    ("hr", "mean"), ("hr", "std"),
]


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_config(name):
    with open(CONFIGS_DIR / f"{name}.yaml") as f:
        return yaml.safe_load(f)


def load_stats():
    if not STATS_PATH.exists():
        raise FileNotFoundError(f"stats.json not found at {STATS_PATH}")
    stats = json.load(open(STATS_PATH))

    for group, key in REQUIRED_STATS_KEYS:
        if group not in stats or key not in stats[group]:
            raise KeyError(f"stats.json missing required key: {group}.{key}")

    if "hr_data_range" not in stats:
        raise KeyError(
            "stats.json missing 'hr_data_range' -- required for PSNR/SSIM/LPIPS "
            "normalization in compute_metrics. Check how stats.json is generated."
        )
    if "hr_percentiles" not in stats or "p1" not in stats.get("hr_percentiles", {}):
        raise KeyError(
            "stats.json missing 'hr_percentiles.p1' -- required as DATA_MIN in "
            "compute_metrics. Check how stats.json is generated."
        )
    return stats


def build_model(stats, cfg, args):
    arch = cfg.get("architecture", {})
    pretrained_cfg = cfg.get("pretrained", {})

    spynet_path = PRETRAINED / pretrained_cfg.get("spynet_filename", "")
    backbone_path = PRETRAINED / pretrained_cfg.get("backbone_filename", "")

    if not spynet_path.exists():
        raise FileNotFoundError(f"SpyNet pretrained weights not found at {spynet_path}")
    if not backbone_path.exists():
        raise FileNotFoundError(f"BasicVSR++ pretrained weights not found at {backbone_path}")

    return BasicVSRModule(
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        data_min=stats["hr_percentiles"]["p1"],

        mid_channels=arch.get("mid_channels", 64),
        num_blocks=arch.get("num_blocks", 7),
        max_residue_magnitude=arch.get("max_residue_magnitude", 10),
        is_low_res_input=arch.get("is_low_res_input", True),
        cpu_cache_length=arch.get("cpu_cache_length", 100),
        spynet_path=str(spynet_path),
        pretrained_path=str(backbone_path),

        lambda_nw=args.lambda_nw,
        lambda_w=args.lambda_w,
        lambda_g=args.lambda_g,
        learning_rate=args.lr,
    )


def build_dataloaders(args):
    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)

    train_ds = SRDataset(split="train", processed_dir=PATCHES_DIR, stats_path=STATS_PATH, augment=True)
    val_ds = SRDataset(split="val", processed_dir=PATCHES_DIR, stats_path=STATS_PATH)
    test_ds = SRDataset(split="test", processed_dir=PATCHES_DIR, stats_path=STATS_PATH)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


def run(args):
    set_seed(args.seed)

    run_name = args.run_name or f"exp_{int(time.time())}"
    stats = load_stats()
    cfg = load_config("basicvsrplus")
    precision = cfg.get("precision", "bf16-mixed")

    train_loader, val_loader, test_loader = build_dataloaders(args)

    # Pretrained weights (backbone + SpyNet) are loaded inside build_model,
    # at construction time -- no separate post-hoc load_pretrained call needed.
    model = build_model(stats, cfg, args)

    logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group,
        config={**cfg, **vars(args)},
    )

    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=run_name + "_epoch={epoch:02d}_val_water_mae={val/water_mae:.4f}",
        monitor="val/water_mae",
        mode="min",
        save_top_k=1,
        auto_insert_metric_name=False,
    )

    callbacks = [
        ckpt,
        EarlyStopping(monitor="val/water_mae", mode="min", patience=args.patience),
        LearningRateMonitor(logging_interval="epoch"),
    ]

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
    trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)

    print(f"[FINISH] Completed in {(time.time() - t0)/60:.2f}m | Best CKPT: {ckpt.best_model_path}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--lambda_nw", type=float, default=1.0)
    p.add_argument("--lambda_w", type=float, default=2.0)
    p.add_argument("--lambda_g", type=float, default=0.1)

    p.add_argument("--project", default="Sequential_TIR")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)
    return p.parse_args()


if __name__ == "__main__":
    run(parse_args())