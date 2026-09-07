import argparse
import json
import random
import numpy as np
import torch
import yaml
import wandb

from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer
from lightning.pytorch.loggers import WandbLogger

from scripts.utils.dataset import SRDataset


# =========================================================
# PATHS
# =========================================================
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
CONFIGS_DIR = BASE / "configs"


# =========================================================
# REPRODUCIBILITY
# =========================================================
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# =========================================================
# CONFIG
# =========================================================
def load_config(model_name):
    path = CONFIGS_DIR / f"{model_name}.yaml"
    with open(path) as f:
        return yaml.safe_load(f)


# =========================================================
# MODEL LOADER
# =========================================================
def get_model_class(model_name):
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule
        return SwinIRModule
    raise ValueError(f"Unknown model: {model_name}")


# =========================================================
# BUILD MODEL FROM CHECKPOINT
# =========================================================
def build_model(args):

    cls = get_model_class(args.model)

    model = cls.load_from_checkpoint(
        args.checkpoint,
        strict=True,
    )

    model.eval()

    return model


# =========================================================
# MAIN
# =========================================================
def run(args):

    set_seed(args.seed)

    # -----------------------
    # LOAD STATS
    # -----------------------
    with open(STATS_PATH) as f:
        stats = json.load(f)

    cfg = load_config(args.model)
    precision = cfg.get("precision", "bf16-mixed")

    USE_AUX = bool(args.use_aux)

    # -----------------------
    # DATASET
    # -----------------------
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

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=True,
    )

    # -----------------------
    # MODEL
    # -----------------------
    model = build_model(args)

    # -----------------------
    # W&B SETUP
    # -----------------------

    strategy = model.hparams.adaptation_strategy

    run_name = args.run_name or f"{args.model}_{strategy}_test"

    wandb_logger = WandbLogger(
        project=args.project,
        name=run_name,
        group=args.group,
        config={
            "strategy": strategy,
            "use_aux": USE_AUX,
            "checkpoint": args.checkpoint,
            "split": "test",
        },
    )

    # -----------------------
    # TRAINER (TEST ONLY)
    # -----------------------
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision=precision,
        logger=wandb_logger,
    )

    # -----------------------
    # TEST
    # -----------------------
    with torch.no_grad():
        results = trainer.test(
            model,
            dataloaders=test_loader
        )

    # -----------------------
    # FINISH W&B
    # -----------------------
    wandb.finish()


# =========================================================
# CLI
# =========================================================
def parse_args():

    p = argparse.ArgumentParser()

    p.add_argument("--model", required=True, choices=["swinir"])
    p.add_argument("--checkpoint", required=True)

    p.add_argument("--use_aux", type=int, default=1)

    p.add_argument("--batch_size", type=int, default=4)
    p.add_argument("--num_workers", type=int, default=4)
    p.add_argument("--seed", type=int, default=42)


    p.add_argument("--project", default="TIR_SISR_EVAL")
    p.add_argument("--group", default="test_eval")
    p.add_argument("--run_name", default=None)


    return p.parse_args()


if __name__ == "__main__":
    args = parse_args()
    run(args)