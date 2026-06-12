# scripts/training/train.py

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
CKPT_DIR = BASE / "checkpoints/gan_v3"
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
    arch.pop("d_lr_scale", None)

    return RealESRGANModule(
        **arch,
        pretrained_path=str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
        pretrained_d_path=str(PRETRAINED / "RealESRGAN_discriminator_x4.pth"),
        learning_rate=args.lr,
        d_lr_scale=args.d_lr_scale,
        aux_chans=aux_chans,
        use_spade=args.use_spade,  
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
    
    run_name = args.run_name or f"exp_{int(time.time())}"
    stats = json.load(open(STATS_PATH))
    cfg = load_config("realesrgan")
    precision = cfg.get("precision", "bf16-mixed")

    # Initialize Datasets
    train_ds = SRDataset(split="train", 
                         patches_dir=PATCHES_DIR, 
                         stats_path=STATS_PATH, 
                         use_aux=True, 
                         use_water_mask=True, 
                         aux_dir=str(PATCHES_DIR / "train" / "AUX"), 
                         transform=train_transforms()
                         )
    
    val_ds = SRDataset(split="val", 
                       patches_dir=PATCHES_DIR, 
                       stats_path=STATS_PATH, 
                       use_aux=True, 
                       use_water_mask=True, 
                       aux_dir=str(PATCHES_DIR / "val" / "AUX"), 
                       transform=None)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                            shuffle=False, **loader_kw)

    # Get auxiliary channel count from the dataset
    raw_aux_chans = train_ds[0]["aux_lr"].shape[0]
    
    # Build Model
    model = build_model(cfg, stats, args, raw_aux_chans)

    logger = WandbLogger(project=args.project, 
                         name=run_name, 
                         group=args.group, 
                         config={**cfg, **vars(args)})

    ckpt = ModelCheckpoint(dirpath=CKPT_DIR, 
                           filename=run_name + "_epoch={epoch:02d}_val_water_mae={val/water_mae:.4f}",
                           monitor="val/water_mae", 
                           mode="min", 
                           save_top_k=1,
                           auto_insert_metric_name=False)

    callbacks = [ckpt, 
                 EarlyStopping(monitor="val/water_mae", 
                               mode="min", 
                               patience=args.patience),
                 LearningRateMonitor(logging_interval="epoch")]

    trainer = Trainer(max_epochs=args.max_epochs, 
                      accelerator="gpu", 
                      devices=1, 
                      precision=precision, 
                      logger=logger, 
                      callbacks=callbacks, 
                      log_every_n_steps=10) 

    print(f"[RUN] {run_name} | SPADE: {model.use_spade}")
    
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)
    
    print(f"[FINISH] Completed in {(time.time() - t0)/60:.2f}m | Best CKPT: {ckpt.best_model_path}")
    wandb.finish()


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--use_spade", type=lambda x: (str(x).lower() == 'true'), default=False)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--d_lr_scale", type=float, default=1.0)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    
    p.add_argument("--lambda_nw", type=float, default=1.0)
    p.add_argument("--lambda_w", type=float, default=1.0)
    p.add_argument("--lambda_g", type=float, default=0.1)
    p.add_argument("--lambda_adversarial", type=float, default=0.1) 
    p.add_argument("--lambda_perceptual", type=float, default=1.0)

    p.add_argument("--project", default="TIR_SISR")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)