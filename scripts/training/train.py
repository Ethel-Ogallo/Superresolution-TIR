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

# Paths
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"
CONFIGS_DIR = BASE / "configs"
CKPT_DIR    = BASE / "checkpoints/dev_gan"
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


# Model loader
def get_model_class(model_name):
    if model_name == "realesrgan":
        from scripts.models.realesrgan import RealESRGANModule
        return RealESRGANModule
    raise ValueError(f"Unknown model: {model_name}")


# Build model
def build_model(cfg, stats, args, aux_chans):
    from scripts.models.realesrgan import RealESRGANModule

    skip = {"model_class", "precision"}
    arch = {k: v for k, v in cfg.items() if k not in skip}

    return RealESRGANModule(
        **arch,
        pretrained_path=str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
        pretrained_d_path=str(PRETRAINED / "RealESRGAN_discriminator_x4.pth"),
        learning_rate=args.lr,
        aux_chans=aux_chans,

        adaptation_strategy=args.adaptation_strategy,
        input_init=args.input_init,
        freeze_backbone=bool(args.freeze_backbone),

        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        data_min=stats["hr_percentiles"]["p1"],
)

# ------------ Training pipeline ----------------
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
    
    test_ds = SRDataset(split="test", 
                        patches_dir=PATCHES_DIR,    
                        stats_path=STATS_PATH,
                        use_aux=True, 
                        use_water_mask=True, 
                        aux_dir=str(PATCHES_DIR / "test" / "AUX"), 
                        transform=None)

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, 
                              shuffle=True, **loader_kw)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, 
                            shuffle=False, **loader_kw)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, 
                             shuffle=False, **loader_kw)
    
    # Get auxiliary channel count from the dataset
    raw_aux_chans = train_ds[0]["aux"].shape[0]
    
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

    
    t0 = time.time()
    trainer.fit(model, train_loader, val_loader)

    # test
    trainer.test(model, test_loader, ckpt_path=ckpt.best_model_path)
    
    print(f"[FINISH] Completed in {(time.time() - t0)/60:.2f}m | Best CKPT: {ckpt.best_model_path}")
    wandb.finish()


# -------------------------
# CLI
# -------------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True, choices=["realesrgan"])
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--max_epochs", type=int, default=100)
    p.add_argument("--patience", type=int, default=20)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--project", default="TIR_sisr_final*")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default=None)
    p.add_argument("--use_aux", type=int, default=1)
    p.add_argument("--adaptation_strategy", type=str, default="projection", choices=["projection", "direct", "fusion"])  
    p.add_argument("--freeze_backbone", type=int, default=0)
    p.add_argument("--freeze_mode",type=str,default="none",choices=["none", "body", "body+first"])
    p.add_argument("--input_init",type=str,default="pretrained_mean",
                   choices=["pretrained_mean","gaussian","xavier","he", "partial_preserve"])

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)