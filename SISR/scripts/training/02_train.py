# scripts/training/train_gan.py
"""
train_realesrgan_aux.py
"""

import argparse
import json
import random
import time
from datetime import timedelta
from pathlib import Path

import numpy as np
import torch
import wandb
import yaml

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

# ----------- paths -----------------
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"
CONFIGS_DIR = BASE / "configs"
CKPT_DIR    = BASE / "checkpoints/dev_gan"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


# ----------- reproducibility -----------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ----------- config -----------------
def load_config(model_name):
    path = CONFIGS_DIR / f"{model_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# ----------- model builder -----------------
def build_model(cfg, stats, args, aux_chans):
    from scripts.models.realesrgan import RealESRGANModule

    skip_keys = {"model_class", "precision"}
    arch_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}

    return RealESRGANModule(
        **arch_kwargs,
        pretrained_path   = str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
        pretrained_d_path = str(PRETRAINED / "RealESRGAN_discriminator_x4.pth"),
        learning_rate     = args.lr,
        aux_chans         = aux_chans,
        adaptation_strategy = args.adaptation_strategy,
        input_init        = args.input_init,
        # lambda_grad       = args.lambda_grad,
        # lambda_water      = args.lambda_water,
        # water_weight      = args.water_weight,
        freeze_backbone   = bool(args.freeze_backbone),
        hr_mean           = stats["hr"]["mean"],
        hr_std            = stats["hr"]["std"],
        data_range        = stats["hr_data_range"],
        data_min          = stats["hr_percentiles"]["p1"],
        # phase             = 3,
    )


# ----------- main -----------------
def run(args):
    set_seed(args.seed)

    print(f"\n{'═'*60}")
    print(f"RealESRGAN — Aux | strategy: {args.adaptation_strategy}")
    print(f"{'═'*60}\n")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg       = load_config("realesrgan")
    precision = cfg.get("precision", "bf16-mixed")

    # ----------- datasets -----------------
    train_ds = SRDataset(
        split          = "train",
        patches_dir    = PATCHES_DIR,
        stats_path     = STATS_PATH,
        use_aux        = True,
        use_water_mask = True,
        aux_dir        = str(PATCHES_DIR / "train" / "AUX"),
        repeat_channels= False,
        transform      = train_transforms(),
    )

    val_ds = SRDataset(
        split          = "val",
        patches_dir    = PATCHES_DIR,
        stats_path     = STATS_PATH,
        use_aux        = True,
        use_water_mask = True,
        aux_dir        = str(PATCHES_DIR / "val" / "AUX"),
        repeat_channels= False,
        transform      = None,
    )

    test_ds = SRDataset(
        split          = "test",
        patches_dir    = PATCHES_DIR,
        stats_path     = STATS_PATH,
        use_aux        = True,
        use_water_mask = True,
        aux_dir        = str(PATCHES_DIR / "test" / "AUX"),
        repeat_channels= False,
        transform      = None,
    )

    loader_kw    = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw) 

    # aux channels 
    sample    = train_ds[0]
    aux_chans = sample["aux"].shape[0]
    print(f"[INFO] AUX channels detected: {aux_chans}")

    # model 
    model = build_model(cfg, stats, args, aux_chans)

    # run name 
    name = args.run_name or (
        f"realesrgan_{args.adaptation_strategy}_aux"
        + (f"_{args.input_init}" if args.adaptation_strategy == "direct" else "")
        + ("_frozen" if args.freeze_backbone else "")
    )

    #  logger 
    wandb_logger = WandbLogger(
        project = args.project,
        name    = name,
        group   = args.group or "phase3_realesrgan_aux",
        config  = {**cfg, **vars(args)},
    )

    # checkpointing 
    ckpt = ModelCheckpoint(
        dirpath    = CKPT_DIR,
        filename   = f"{name}_{{epoch:02d}}_{{val/water_mae:.4f}}",
        monitor    = "val/water_mae",
        mode       = "min",
        save_top_k = 1,
        verbose    = True,
    )

    callbacks = [
        ckpt,
        EarlyStopping(
            monitor = "val/water_mae",
            mode    = "min",
            patience= args.patience,
            verbose = True,
        ),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # ----------- trainer -----------------
    trainer = Trainer(
        max_epochs        = args.max_epochs,
        accelerator       = "gpu",
        devices           = 1,
        precision         = precision,
        logger            = wandb_logger,
        callbacks         = callbacks,
        log_every_n_steps = 5,
        num_sanity_val_steps = 0,
    )

    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    print(f"\n[INFO] Training time: {timedelta(seconds=int(time.time()-t0))}")
    print(f"[INFO] Best checkpoint: {ckpt.best_model_path}")
    print(f"[INFO] Best val/water_mae: {ckpt.best_model_score:.4f}")
    
    trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)
    
    wandb.finish()
    return ckpt.best_model_path


# ----------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--adaptation_strategy", type=str, default="projection",
                   choices=["projection", "direct"])
    p.add_argument("--input_init", type=str, default="pretrained_mean",
                   choices=["pretrained_mean", "gaussian", "xavier", "he", "partial_preserve"])
    p.add_argument("--freeze_backbone", type=int, default=1,
                   help="1=freeze generator backbone (recommended for stability)")
    p.add_argument("--lr",          type=float, default=1e-4)
    p.add_argument("--batch_size",  type=int,   default=4)
    p.add_argument("--max_epochs",  type=int,   default=100)
    p.add_argument("--patience",    type=int,   default=20)
    p.add_argument("--num_workers", type=int,   default=4)
    p.add_argument("--seed",        type=int,   default=42)
    p.add_argument("--lambda_grad",  type=float, default=0.0)
    p.add_argument("--lambda_water", type=float, default=0.0)
    p.add_argument("--water_weight", type=float, default=1.0)
    p.add_argument("--use_aux", type=int, default=1)
    p.add_argument("--project",    default="TIR_SISR")
    p.add_argument("--run_name",   default=None)
    p.add_argument("--group",      default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)