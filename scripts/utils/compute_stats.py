"""
compute_stats.py — Compute global normalisation statistics from train tiles only.
Saves stats.json to the patches directory.
"""

import json
import numpy as np
from pathlib import Path

# Paths 
PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
TRAIN_HR    = PATCHES_DIR / "train" / "HR"
TRAIN_LR    = PATCHES_DIR / "train" / "LR"
STATS_PATH  = PATCHES_DIR / "stats.json"

#  Compute stats from train tiles only 
def compute_stats(tile_dir):
    """Compute mean and std from all .npy tiles in a directory."""
    files = sorted(tile_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {tile_dir}")

    n    = 0
    s    = 0.0
    ss   = 0.0
    vmin = np.inf
    vmax = -np.inf

    for f in files:
        tile = np.load(f).astype(np.float64)
        valid = tile[np.isfinite(tile)]
        if len(valid) == 0:
            continue
        n    += len(valid)
        s    += valid.sum()
        ss   += (valid ** 2).sum()
        vmin  = min(vmin, valid.min())
        vmax  = max(vmax, valid.max())

    if n == 0:
        raise ValueError(f"No valid pixels found in {tile_dir}")

    mean = s / n
    std  = np.sqrt(ss / n - mean ** 2)

    return {
        "mean": round(float(mean), 6),
        "std":  round(float(std),  6),
        "min":  round(float(vmin), 6),
        "max":  round(float(vmax), 6),
        "n_valid_pixels": int(n)
    }

def compute_data_range(tile_dir, low_p=1.0, high_p=99.0):
    """Compute data range from percentile of valid pixels (train only)."""
    files = sorted(tile_dir.glob("*.npy"))
    if not files:
        raise FileNotFoundError(f"No .npy files found in {tile_dir}")

    values = []

    for f in files:
        tile = np.load(f).astype(np.float64)
        valid = tile[np.isfinite(tile)]
        if valid.size > 0:
            values.append(valid)

    if len(values) == 0:
        raise ValueError(f"No valid pixels in {tile_dir}")

    values = np.concatenate(values)

    low  = np.percentile(values, low_p)
    high = np.percentile(values, high_p)

    return {
        "low_percentile": float(low),
        "high_percentile": float(high),
        "data_range": float(high - low)
    }


print("Computing HR stats from train tiles...")
hr_stats = compute_stats(TRAIN_HR)
print(f"  mean={hr_stats['mean']:.4f}  std={hr_stats['std']:.4f}  "
      f"min={hr_stats['min']:.4f}  max={hr_stats['max']:.4f}  "
      f"n={hr_stats['n_valid_pixels']:,}")

print("Computing LR stats from train tiles...")
lr_stats = compute_stats(TRAIN_LR)
print(f"  mean={lr_stats['mean']:.4f}  std={lr_stats['std']:.4f}  "
      f"min={lr_stats['min']:.4f}  max={lr_stats['max']:.4f}  "
      f"n={lr_stats['n_valid_pixels']:,}")

print("Computing HR data range (train tiles)...")
hr_range = compute_data_range(TRAIN_HR)

print(f"  p1={hr_range['low_percentile']:.4f}  "
      f"p99={hr_range['high_percentile']:.4f}  "
      f"range={hr_range['data_range']:.4f}")

# Save stats
stats = {
    "description": "Global normalisation stats computed from train tiles only",
    "hr": hr_stats,
    "lr": lr_stats,
    "hr_data_range": hr_range["data_range"],
    "hr_percentiles": {
        "p1": hr_range["low_percentile"],
        "p99": hr_range["high_percentile"]
    }
}
with open(STATS_PATH, "w") as f:
    json.dump(stats, f, indent=2)

print(f"\nStats saved to {STATS_PATH}")