import json
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

# ── paths ──────────────────────────────────────────────
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
CKPT        = "checkpoints/dev_v2/swinir/best-epoch=90-val_full_psnr=18.2759.ckpt"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ── load stats ─────────────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)

HR_MEAN      = stats["hr"]["mean"]        # 30.34
HR_STD       = stats["hr"]["std"]         # 5.89
DATA_RANGE   = stats["hr_data_range"]     # 27.59
DATA_MIN     = stats["hr_percentiles"]["p1"]

# ── load model ─────────────────────────────────────────
model = SwinIRModule.load_from_checkpoint(CKPT, strict=True)
model.eval().to(device)

print(f"model.hr_mean = {model.hr_mean}")
print(f"model.hr_std  = {model.hr_std}")

# ── dataset ────────────────────────────────────────────
test_ds = SRDataset(
    split="test",
    patches_dir=PATCHES_DIR,
    stats_path=STATS_PATH,
    use_aux=True,
    use_water_mask=True,
    aux_dir=str(PATCHES_DIR / "test" / "AUX"),
    repeat_channels=False,
    transform=None,
)

loader = DataLoader(test_ds, batch_size=4, shuffle=False, num_workers=2)

# ── inference loop ─────────────────────────────────────
psnr_list, mae_list = [], []

with torch.no_grad():
    for batch in loader:
        lr      = batch["lr"].to(device)
        hr      = batch["hr"].to(device)
        hr_mask = batch["hr_mask"].to(device)
        b_in    = {"lr": lr}
        if "aux" in batch:
            b_in["aux"] = batch["aux"].to(device)

        sr_norm = model(b_in)

        # denormalize to °C
        sr = sr_norm[:, 0:1] * HR_STD + HR_MEAN
        hr = hr[:, 0:1]      * HR_STD + HR_MEAN

        print(f"sr range: {sr.min():.2f} to {sr.max():.2f}")
        print(f"hr range: {hr.min():.2f} to {hr.max():.2f}")

        mask = hr_mask > 0.5
        sr_v = sr[mask]
        hr_v = hr[mask]

        mse  = ((sr_v - hr_v) ** 2).mean()
        psnr = 10 * torch.log10(DATA_RANGE ** 2 / (mse + 1e-8))
        mae  = torch.abs(sr_v - hr_v).mean()

        psnr_list.append(psnr.item())
        mae_list.append(mae.item())

print(f"\nMean PSNR : {np.mean(psnr_list):.4f} dB")
print(f"Mean MAE  : {np.mean(mae_list):.4f} °C")