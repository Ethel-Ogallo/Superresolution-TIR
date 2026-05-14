# scripts/training/train.py

import argparse
import json
import random
import numpy as np
import torch
import wandb
import yaml
import time
from datetime import timedelta

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


# Model loader
def get_model_class(model_name):
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    raise ValueError(f"Unknown model: {model_name}")


# Build model
def build_model(model_name, cfg, stats, args, in_aux_chans=None):  

    cls = get_model_class(model_name)
    pretrained_path = str(PRETRAINED / "SwinIR_classical_x4.pth")
    skip_keys = {"model_class", "precision"}
    arch_kwargs = {k: v for k, v in cfg.items() if k not in skip_keys}

    return cls(
        **arch_kwargs,
        pretrained_path=pretrained_path,
        learning_rate=args.lr,
        in_aux_chans=in_aux_chans,
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats["hr_data_range"],
        adaptation_strategy=args.adaptation_strategy,
        lambda_grad=args.lambda_grad,    
        lambda_water=args.lambda_water,  
    )

# ------------ Training pipeline ----------------
def run(args):

    set_seed(args.seed)

    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg = load_config(args.model)
    precision = cfg.get("precision", "bf16-mixed")

    USE_AUX = bool(args.use_aux)

    transforms = train_transforms() if USE_AUX else None

    # DATASETS + DATALOADERS
    train_ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "train" / "AUX") if USE_AUX else None,
        repeat_channels=False if USE_AUX else True,
        transform=transforms,
    )

    val_ds = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "val" / "AUX") if USE_AUX else None,
        repeat_channels=False if USE_AUX else True,
        transform=None,
    )

    test_ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,   
        aux_dir=str(PATCHES_DIR / "test" / "AUX") if USE_AUX else None,
        repeat_channels=False if USE_AUX else True,
        transform=None,
    )

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)


    # MODEL
    if USE_AUX:
        sample = train_ds[0]
        in_aux_chans = sample["aux"].shape[0]
        print(f"[INFO] AUX channels detected: {in_aux_chans}")
    else:
        in_aux_chans = None

    # MODEL
    model = build_model(args.model, cfg, stats, args, in_aux_chans)

    # LOGGER
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or f"{args.model}_aux{int(USE_AUX)}",
        group=args.group,
        config={**cfg, **vars(args)},
    )


    # CHECKPOINTING
    ckpt = ModelCheckpoint(
        dirpath=CKPT_DIR / args.model,
        filename="best-{epoch:02d}-{val_full_psnr:.4f}",
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


    # TRAINER
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision=precision,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=5,
    )

    # TRAIN
    train_start = time.time()

    trainer.fit(model, train_loader, val_loader)

    train_end = time.time()
    train_time = timedelta(seconds=int(train_end - train_start))

    print(f"\n[INFO] Total training time: {train_time}")

    best_ckpt = ckpt.best_model_path
    print("\n[INFO] Best checkpoint:", best_ckpt)

    # TEST
    # test_results = trainer.test(
    #     model,
    #     dataloaders=test_loader,
    #     ckpt_path="best"
    # )

    wandb.finish()

    return best_ckpt


# -------------------------
# CLI
# -------------------------
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
    p.add_argument("--use_aux", type=int, default=1)
    p.add_argument("--adaptation_strategy", type=str, default="projection",choices=["projection", "direct"])
    p.add_argument("--lambda_grad", type=float, default=0.0)   
    p.add_argument("--lambda_water", type=float, default=0.0)  

    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)