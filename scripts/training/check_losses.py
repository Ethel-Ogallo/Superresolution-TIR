# check_losses.py
import json
import torch
import torch.nn.functional as F
from pathlib import Path
from torch.utils.data import DataLoader

from kornia.filters import SpatialGradient

import sys

PROJECT_ROOT = Path(__file__).parent.parent.parent  
sys.path.append(str(PROJECT_ROOT))

print(f"Project root added to path: {PROJECT_ROOT}\n")
from scripts.utils.dataset import SRDataset

BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"

ds = SRDataset(
    split="val",
    patches_dir=PATCHES_DIR,
    stats_path=STATS_PATH,
    use_aux=True,
    use_water_mask=True,
    aux_dir=str(PATCHES_DIR / "val" / "AUX"),
    repeat_channels=False,
    transform=None,
)

loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)

# ---- accumulators ----
total_valid        = 0
total_water        = 0
total_strict_inner = 0
total_relaxed_inner = 0
n_batches          = 0

for batch in loader:
    hr_mask    = batch["hr_mask"]    # [B, 1, H, W]
    water_mask = batch.get("water_mask", None)

    # --- CHECK 1: water pixel fraction ---
    valid_pixels = hr_mask.sum().item()
    total_valid += valid_pixels

    if water_mask is not None:
        water_pixels = (hr_mask * water_mask).sum().item()
        total_water += water_pixels

    # --- CHECK 2: gradient loss mask comparison ---
    strict_inner  = (F.avg_pool2d(hr_mask, kernel_size=3, stride=1, padding=1) > 0.99).float()
    relaxed_inner = (F.avg_pool2d(hr_mask, kernel_size=3, stride=1, padding=1) > 0.50).float()

    total_strict_inner  += strict_inner.sum().item()
    total_relaxed_inner += relaxed_inner.sum().item()

    n_batches += 1

# ---- results ----
print("\n" + "="*50)
print("  WATER LOSS CHECK")
print("="*50)
water_frac = total_water / max(total_valid, 1)
print(f"  Total valid pixels : {int(total_valid)}")
print(f"  Total water pixels : {int(total_water)}")
print(f"  Water fraction     : {water_frac:.4f}  ({water_frac*100:.1f}%)")

if water_frac < 0.05:
    print("  → VERDICT: water loss barely firing (<5%) — not worth λ=0.5 weight")
elif water_frac < 0.10:
    print("  → VERDICT: water loss firing occasionally — modest impact")
else:
    print("  → VERDICT: water loss firing regularly — meaningful contribution")

print("\n" + "="*50)
print("  GRADIENT LOSS MASK CHECK")
print("="*50)
strict_frac  = total_strict_inner  / max(total_valid, 1)
relaxed_frac = total_relaxed_inner / max(total_valid, 1)
print(f"  Strict  (>0.99) pixels kept : {strict_frac:.4f}  ({strict_frac*100:.1f}% of valid)")
print(f"  Relaxed (>0.50) pixels kept : {relaxed_frac:.4f}  ({relaxed_frac*100:.1f}% of valid)")
print(f"  Signal lost by strict mask  : {(relaxed_frac - strict_frac)*100:.1f}%")

if (relaxed_frac - strict_frac) > 0.20:
    print("  → VERDICT: strict mask is throwing away >20% of signal — relax it")
else:
    print("  → VERDICT: strict mask is not the bottleneck — threshold is fine")

print("="*50)