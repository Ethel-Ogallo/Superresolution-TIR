"""
train.py — TIR SR training with stratified random split.
Works across EDSR / SwinIR / HAT / RealESRGAN.
Safely handles AUX + FiLM without breaking non-AUX models.

Fixes applied:
  Bug 3 — aux_channels (total detected from dataset) is now always passed
           when use_aux=True, so the model never falls back to a stale hparam sum.
"""

import argparse
import json
import os
import random

import numpy as np
import pandas as pd
import torch
import wandb
import yaml

from lightning.pytorch import Trainer
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import WandbLogger
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from scripts.utils.dataset import (
    SRDataset,
    compute_mean_std,
    compute_aux_mean_std,
    compute_data_range,
    Compose,
    GeoAugment,
    TIRNoise,
    BlurAugment,
)


# ---------------------- Reproducibility ---------------------
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ---------------------- Config ---------------------
def load_model_config(model_name):
    repo_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    path = os.path.join(repo_root, "configs", f"{model_name}.yaml")
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------- Model registry ---------------------
def get_model_class(model_name):
    if model_name == "edsr":
        from scripts.models.edsr import EDSRModule
        return EDSRModule
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    if model_name == "hat":
        from scripts.models.hat import HATModule
        return HATModule
    if model_name == "real_esrgan":
        from scripts.models.real_esrgan import RealESRGANModule
        return RealESRGANModule
    raise ValueError(f"Unknown model: {model_name}")


# ---------------------- Split ---------------------
def stratified_random_split(metadata):
    train_val, test_df = train_test_split(
        metadata, test_size=0.15, random_state=42,
        stratify=metadata["campaign_id"],
    )
    train_df, val_df = train_test_split(
        train_val, test_size=0.15 / 0.85, random_state=42,
        stratify=train_val["campaign_id"],
    )
    print(f"[INFO] Train {len(train_df)} | Val {len(val_df)} | Test {len(test_df)}")
    return train_df, val_df, test_df


def save_split(train_df, val_df, test_df, metadata_json):
    with open(metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))

    metadata["split"] = None
    metadata.loc[metadata["patch_name"].isin(train_df["patch_name"]), "split"] = "train"
    metadata.loc[metadata["patch_name"].isin(val_df["patch_name"]),   "split"] = "val"
    metadata.loc[metadata["patch_name"].isin(test_df["patch_name"]),  "split"] = "test"

    with open(metadata_json, "w") as f:
        json.dump(json.loads(metadata.to_json(orient="records", indent=2)), f, indent=2)


# ---------------------- TRAIN ---------------------
def run(train_df, val_df, test_df, args, model_cfg):

    print("\n===== MODEL:", args.model.upper(), "=====")

    mean, std = compute_mean_std(train_df)
    print(f"[INFO] Train HR mean: {mean:.4f} | std: {std:.4f}")

    data_range = compute_data_range(train_df)
    print(f"[INFO] Train data range: {data_range:.4f}")

    # ---------------- AUX ----------------
    use_aux = args.use_aux
    print(f"[INFO] use_aux = {use_aux}")

    aux_mean, aux_std = None, None
    if use_aux:
        aux_mean, aux_std = compute_aux_mean_std(train_df)
        print("[INFO] AUX normalization enabled")

    train_aug = Compose([
        GeoAugment(),
        TIRNoise(std=std, p=0.5),
        BlurAugment(),
    ])

    train_ds = SRDataset(
        train_df, mean=mean, std=std,
        aux_mean=aux_mean, aux_std=aux_std,
        patch_size=model_cfg.get("patch_size", 48),
        is_train=True, transforms=train_aug,
        use_aux=use_aux,
    )
    val_ds = SRDataset(
        val_df, mean=mean, std=std,
        aux_mean=aux_mean, aux_std=aux_std,
        is_train=False, transforms=None,
        use_aux=use_aux,
    )
    test_ds = SRDataset(
        test_df, mean=mean, std=std,
        aux_mean=aux_mean, aux_std=aux_std,
        is_train=False, transforms=None,
        use_aux=use_aux,
    )

    loader_kw = dict(num_workers=args.num_workers, pin_memory=True)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,  **loader_kw)
    val_loader   = DataLoader(val_ds,   batch_size=args.batch_size, shuffle=False, **loader_kw)
    test_loader  = DataLoader(test_ds,  batch_size=1,               shuffle=False, **loader_kw)

    # Detect total aux channels from an actual sample (cont bands + one-hot LULC)
    # This is the single source of truth passed into every model.
    aux_channels = None
    if use_aux:
        aux_channels = train_ds[0][-1].shape[0]  # aux_tensor.shape[0]
        print(f"[INFO] detected aux_channels = {aux_channels}")

    # ---------------- MODEL ----------------
    cls = get_model_class(args.model)

    # Keys handled explicitly below — strip from yaml to avoid duplicate kwargs
    explicit_keys = {
        "learning_rate", "patience", "precision",
        "mean", "std", "data_range",
        "aux_mean", "aux_std",
        "pretrained_path",
        "freeze_backbone", "bb_lr_scale", "lambda_grad",
        "model_class",
        "use_aux",
        "aux_channels",       # total count — always passed from here
        "aux_cont_channels",  # fallback inside model; don't override from yaml
        "num_lulc_classes",
        "patch_size",
        "fusion_mode",
    }
    model_hparams = {k: v for k, v in model_cfg.items() if k not in explicit_keys}

    model_kwargs = dict(
        mean=mean,
        std=std,
        data_range=data_range,
        learning_rate=args.lr,
        patience=args.patience,
        pretrained_path=args.pretrained,
        freeze_backbone=args.freeze_backbone,
        bb_lr_scale=args.bb_lr_scale,
        lambda_grad=args.lambda_grad,
        use_aux=use_aux,
        fusion_mode=args.fusion_mode,
        **model_hparams,
    )

    #  always pass the detected total to any model that accepts aux_channels
    sig = cls.__init__.__code__.co_varnames
    if use_aux and aux_channels is not None and "aux_channels" in sig:
        model_kwargs["aux_channels"] = aux_channels

    # Pass aux stats if the model signature supports them
    if "aux_mean" in sig:
        model_kwargs["aux_mean"] = aux_mean
    if "aux_std" in sig:
        model_kwargs["aux_std"] = aux_std

    model = cls(**model_kwargs)

    # ---------------- W&B ----------------
    wandb_logger = WandbLogger(
        project=args.project,
        name=args.run_name or args.model,
        group=args.group or args.model,
        log_model="all",
        config={**model_cfg, **vars(args)},
    )

    callbacks = [
        EarlyStopping(monitor="val_psnr", mode="max", patience=args.patience),
        ModelCheckpoint(monitor="val_psnr", mode="max", save_top_k=1),
        LearningRateMonitor(logging_interval="epoch"),
    ]

    trainer = Trainer(
        max_epochs=args.max_epochs,
        accelerator="auto",
        devices="auto",
        precision=model_cfg.get("precision", "bf16-mixed"),
        logger=wandb_logger,
        callbacks=callbacks,
    )

    trainer.fit(model, train_loader, val_loader)
    trainer.test(model, test_loader, ckpt_path="best")

    wandb.finish()


# ---------------------- MAIN ---------------------
def main():
    args = parse_args()
    set_seed(args.seed)

    model_cfg = load_model_config(args.model)

    with open(args.metadata_json) as f:
        metadata = pd.DataFrame(json.load(f))

    if "campaign_id" not in metadata.columns:
        metadata["campaign_id"] = metadata["patch_name"]

    train_df, val_df, test_df = stratified_random_split(metadata)
    save_split(train_df, val_df, test_df, args.metadata_json)

    run(train_df, val_df, test_df, args, model_cfg)


# ---------------------- CLI ---------------------
def parse_args():
    p = argparse.ArgumentParser()

    p.add_argument("--model",          required=True)
    p.add_argument("--metadata_json",  default="data/full_metadata.json")
    p.add_argument("--pretrained",     default=None)

    p.add_argument("--lr",             type=float, default=1e-4)
    p.add_argument("--batch_size",     type=int,   default=4)
    p.add_argument("--max_epochs",     type=int,   default=100)
    p.add_argument("--patience",       type=int,   default=20)

    p.add_argument("--num_workers",    type=int,   default=2)
    p.add_argument("--seed",           type=int,   default=42)

    p.add_argument("--freeze_backbone", action="store_true")
    p.add_argument("--bb_lr_scale",    type=float, default=1.0)
    p.add_argument("--lambda_grad",    type=float, default=0.5)

    p.add_argument("--use_aux",        action="store_true")
    p.add_argument("--fusion_mode", default="film", choices=["film", "concat"])

    p.add_argument("--project",        default="TIR_sisr")
    p.add_argument("--run_name",       default=None)
    p.add_argument("--group",          default=None)

    return p.parse_args()


if __name__ == "__main__":
    main()