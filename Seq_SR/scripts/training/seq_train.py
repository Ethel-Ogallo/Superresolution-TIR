# scripts/training/seq_train.py
# spynet frozen

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

from scripts.utils.seq_dataset import SRSequenceDataset
from scripts.models.basicvsrplus import BasicVSRModule


BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/seq_patches2"
STATS_PATH = BASE / "data/processed/patches/stats.json"
DEFAULT_CONFIG_PATH = BASE / "configs/basicvsrplus.yaml"
CKPT_DIR = BASE / "checkpoints/vsr"
CKPT_DIR.mkdir(parents=True, exist_ok=True)


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def load_stats():
    with open(STATS_PATH) as f:
        return json.load(f)


def load_config(config_path):
    path = Path(config_path)
    if not path.exists():
        print(f"[WARNING] Config file not found at {path}. Proceeding without YAML config.")
        return {}
    with open(path, "r") as f:
        return yaml.safe_load(f)


def build_model(stats, args, config):
    pretrained_dir = BASE / "data/pretrained"
    pretrained_cfg = config.get("pretrained", {})
    backbone_file = pretrained_cfg.get("backbone_filename")
    spynet_file = pretrained_cfg.get("spynet_filename")
    pretrained_path = (str(pretrained_dir / backbone_file) if backbone_file else None )
    spynet_path = (str(pretrained_dir / spynet_file) if spynet_file else None )

    return BasicVSRModule(
        hr_mean=stats["hr"]["mean"],
        hr_std=stats["hr"]["std"],
        data_range=stats.get("hr_data_range"),
        data_min=stats.get("hr_percentiles",{}).get("p1"),
        mid_channels=config.get("mid_channels", 64),
        num_blocks=config.get("num_blocks", 7),
        learning_rate=args.lr,
        pretrained_path=pretrained_path,
        spynet_path=spynet_path,
        use_aux=args.use_aux,
        n_aux_channels=args.n_aux_channels,
        use_spade=args.use_spade,        
        seg_nc=args.seg_nc,              
        lambda_nw=args.lambda_nw,
        lambda_w=args.lambda_w,
        lambda_g=args.lambda_g,
    )

def build_dataloaders(args):
    loader_kw = {"num_workers": args.num_workers, "pin_memory": True}

    train_ds = SRSequenceDataset("train", PATCHES_DIR, STATS_PATH)
    val_ds = SRSequenceDataset("val", PATCHES_DIR, STATS_PATH)
    test_ds = SRSequenceDataset("test", PATCHES_DIR, STATS_PATH)

    train_loader = torch.utils.data.DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, **loader_kw)
    val_loader = torch.utils.data.DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader = torch.utils.data.DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, **loader_kw)

    return train_loader, val_loader, test_loader


def run(args):
    set_seed(args.seed)
    stats = load_stats()
    config = load_config(args.config)

    train_loader, val_loader, test_loader = build_dataloaders(args)
    model = build_model(stats, args, config)

    run_name = args.run_name or f"basicvsrpp_baseline_ep{args.max_epochs}"

    logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group,
        save_dir="wandb_logs",
    )

    ckpt_callback = ModelCheckpoint(
        dirpath=CKPT_DIR,
        filename=f"{run_name}_epoch={{epoch:02d}}_val_water_mae={{val/water_mae:.4f}}",
        monitor="val/water_mae",
        mode="min",
        save_top_k=1,
        save_last=True,
    )

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="gpu",
        devices=1,
        precision="bf16-mixed",
        logger=logger,
        callbacks=[
            ckpt_callback,
            EarlyStopping(monitor="val/water_mae", patience=args.patience, mode="min"),
            LearningRateMonitor(logging_interval="epoch"),
        ],
        log_every_n_steps=10,
        accumulate_grad_batches=args.accumulate_grad_batches,
        num_sanity_val_steps=2,
    )

    print(f"Starting training: {run_name} | Group: {args.group}")
    trainer.fit(model, train_loader, val_loader)

    if ckpt_callback.best_model_path:
        print(f"Testing best checkpoint: {ckpt_callback.best_model_path}")
        trainer.test(model, test_loader, ckpt_path=ckpt_callback.best_model_path)

    wandb.finish()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    
    # Path to YAML config
    p.add_argument("--config", type=str, default=str(DEFAULT_CONFIG_PATH))


    p.add_argument("--batch_size", type=int, default=2)
    p.add_argument("--accumulate_grad_batches", type=int, default=4)
    p.add_argument("--use_aux", action="store_true")             
    p.add_argument("--n_aux_channels", type=int, default=25)   
    p.add_argument("--use_spade", action="store_true")     
    p.add_argument("--seg_nc", type=int, default=14)          

    # Hyperparameters
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--flow_lr", type=float, default=0.0)
    p.add_argument("--max_epochs", type=int, default=50)
    p.add_argument("--patience", type=int, default=10)
    p.add_argument("--num_workers", type=int, default=6)
    p.add_argument("--seed", type=int, default=42)

    # Loss weights
    p.add_argument("--lambda_nw", type=float, default=1.0)
    p.add_argument("--lambda_w", type=float, default=2.0)
    p.add_argument("--lambda_g", type=float, default=0.1)

    # Logging
    p.add_argument("--project", default="Sequential_TIR")
    p.add_argument("--group", default="baseline")
    p.add_argument("--run_name", default="None")

    args = p.parse_args()
    run(args)