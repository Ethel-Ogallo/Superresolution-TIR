import json
import argparse
import torch
import numpy as np
from pathlib import Path
from torch.utils.data import DataLoader

import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent  
sys.path.append(str(PROJECT_ROOT))

print(f"Project root added to path: {PROJECT_ROOT}\n")

from scripts.utils.dataset import SRDataset
from scripts.models.swinir import SwinIRModule

# -----------------------------
# CONFIG
# -----------------------------
BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH = PATCHES_DIR / "stats.json"
PRETRAINED = BASE / "data/pretrained" / "SwinIR_classical_x4.pth"


# -----------------------------
# LOAD DATASET
# -----------------------------
ds = SRDataset(
    split="train",
    patches_dir=PATCHES_DIR,
    stats_path=STATS_PATH,
    use_aux=True,
    use_water_mask=True,
    aux_dir=str(PATCHES_DIR / "train" / "AUX"),
    repeat_channels=False,
    transform=None,
)

loader = DataLoader(ds, batch_size=1, shuffle=False)


# -----------------------------
# GET ONE SAMPLE
# -----------------------------
batch = next(iter(loader))

print("\n========== DATASET SHAPES ==========")
for k, v in batch.items():
    if torch.is_tensor(v):
        print(f"{k:15s} {tuple(v.shape)} {v.dtype}")
    else:
        print(f"{k:15s} {type(v)}")


# -----------------------------
# DETECT AUX CHANNELS
# -----------------------------
aux_chans = batch["aux_mid"].shape[1] if "aux_mid" in batch else None
print("\nDetected aux channels:", aux_chans)


# -----------------------------
# LOAD MODEL (DIRECT SPADE VERSION)
# -----------------------------
model = SwinIRModule(
    pretrained_path=str(PRETRAINED),
    aux_chans=aux_chans,
    lulc_channels=[0],   # change if needed
    hr_mean=0,
    hr_std=1,
    freeze_backbone=False,
)

model.eval()


# -----------------------------
# FORWARD PASS DEBUG
# -----------------------------
with torch.no_grad():

    print("\n========== FORWARD DEBUG ==========")

    out = model(batch)

    print("\n========== OUTPUT ==========")
    print("output shape:", tuple(out.shape))


# -----------------------------
# EXTRA SANITY CHECKS
# -----------------------------
print("\n========== SANITY CHECKS ==========")

lr = batch["lr"]
hr = batch["hr"]

print("LR shape:", lr.shape)
print("HR shape:", hr.shape)

scale = hr.shape[-1] // lr.shape[-1]
print("Upscale factor inferred:", scale)

assert scale in [2, 4, 8], "Unexpected scale mismatch"

print("\n✔ Debug completed successfully")