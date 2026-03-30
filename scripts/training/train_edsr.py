"""
train_edsr.py — Train the EDSR model for TIR super-resolution.

Launch via SLURM: sbatch bash/train_edsr.sh
Edit hyperparameters directly in the .sh file.

"""

import argparse
import os
import random

import numpy as np
import pandas as pd
import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

import wandb
import lightning.pytorch as pl
from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger

from scripts.utils.dataset import SRDataset, compute_mean_std
from scripts.models.edsr import EDSRModule

# -------------------- Reproducibility --------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# -------------------- Data splits --------------------
def build_splits(metadata_csv: str):
    metadata = pd.read_csv(metadata_csv)

    # Recompute splits if missing
    if "split" not in metadata.columns or metadata["split"].isna().all():
        train_val, test = train_test_split(metadata, test_size=0.15, random_state=42)
        train, val = train_test_split(train_val, test_size=0.15 / 0.85, random_state=42)
        metadata["split"] = None
        metadata.loc[train.index, "split"] = "train"
        metadata.loc[val.index, "split"] = "val"
        metadata.loc[test.index, "split"] = "test"
        metadata.to_csv(metadata_csv, index=False)
        print(f"Splits written back to {metadata_csv}")

    train = metadata[metadata["split"] == "train"]
    val = metadata[metadata["split"] == "val"]
    test = metadata[metadata["split"] == "test"]
    print(f"Split sizes — train: {len(train)}, val: {len(val)}, test: {len(test)}")
    return train, val, test


# -------------------- Main training --------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

    # Dataset splits & statistics
    train_df, val_df, test_df = build_splits(args.metadata_csv)
    mean, std = compute_mean_std(train_df)

    train_ds = SRDataset(train_df, do_augment=True, mean=mean, std=std)
    val_ds = SRDataset(val_df, do_augment=False, mean=mean, std=std)
    test_ds = SRDataset(test_df, do_augment=False, mean=mean, std=std)

    loader_kwargs = dict(batch_size=args.batch_size, num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, shuffle=True, **loader_kwargs)
    val_loader = DataLoader(val_ds, shuffle=False, **loader_kwargs)
    test_loader = DataLoader(test_ds, shuffle=False, batch_size=1,
                             num_workers=args.num_workers, pin_memory=True)

    # Model init
    model = EDSRModule(
        pretrained_path=args.pretrained,
        mean=mean,
        std=std,
        learning_rate=args.lr,
        backbone_lr_scale=args.bb_lr_scale,
        patience=args.patience,
        n_feats=args.n_feats,
        n_blocks=args.n_blocks,
        freeze_backbone=not args.unfreeze,
    )

    # Run name
    frozen_tag = "finetune" if args.unfreeze else "frozen"
    run_name = args.run_name or f"EDSR_{frozen_tag}_nf{args.n_feats}_lr{args.lr:.0e}"

    # W&B logger
    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        log_model="all",
        config=vars(args),
    )

    # Callbacks
    callbacks = [
        EarlyStopping(monitor="val_psnr", patience=args.patience, mode="max", verbose=True),
        ModelCheckpoint(monitor="val_psnr", mode="max", save_top_k=1,
                        filename=f"{run_name}-{{epoch:03d}}-{{val_psnr:.2f}}", dirpath=None),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    # Trainer
    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision="16-mixed",
        gradient_clip_val=1.0,
        logger=wandb_logger,
        callbacks=callbacks,
        log_every_n_steps=1,
    )

    # Minimal terminal print
    print(f"Starting run: {run_name}")

    # Training
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

    # Optional testing (default: off for large datasets)
    if args.run_test:
        print("Evaluating on test set …")
        trainer.test(model, dataloaders=test_loader, ckpt_path="best")

    wandb.finish()
    print("Done.")


# -------------------- CLI args --------------------
def parse_args():
    p = argparse.ArgumentParser(description="EDSR baseline training")

    # Data
    p.add_argument("--metadata_csv", default="metadata.csv")
    # Model
    p.add_argument("--pretrained", default=None)
    p.add_argument("--n_feats", type=int, default=64)
    p.add_argument("--n_blocks", type=int, default=16)
    p.add_argument("--unfreeze", action="store_true", help="Unfreeze full backbone (default: freeze, train head only)")

    # Training
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--bb_lr_scale", type=float, default=0.1)
    p.add_argument("--max_epochs", type=int, default=20)
    p.add_argument("--batch_size", type=int, default=1)
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run_test", action="store_true", help="Run full test after training (default: off)")

    # W&B
    p.add_argument("--project", default="TIR_sisr")
    p.add_argument("--run_name", default=None)

    return p.parse_args()


if __name__ == "__main__":
    main()