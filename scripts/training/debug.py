# debug_check.py
# Run with: python debug_check.py --adaptation_strategy projection
# No training required — just one forward pass

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

# =========================================================
# PATHS  (mirrors train.py)
# =========================================================
BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"


# =========================================================
# HELPERS
# =========================================================
def sep(title=""):
    print("\n" + "="*60)
    if title:
        print(f"  {title}")
        print("="*60)


def tensor_stats(name, t):
    t = t.float()
    print(f"  {name:20s}  shape={tuple(t.shape)}  "
          f"min={t.min():.4f}  max={t.max():.4f}  "
          f"mean={t.mean():.4f}  std={t.std():.4f}  "
          f"nan={t.isnan().sum().item()}  inf={t.isinf().sum().item()}")


# =========================================================
# MAIN
# =========================================================
def main(args):

    # ---- Stats ----
    with open(STATS_PATH) as f:
        stats = json.load(f)

    hr_mean      = stats["hr"]["mean"]          # 30.342
    hr_std       = stats["hr"]["std"]           # 5.889
    lr_mean      = stats["lr"]["mean"]
    lr_std       = stats["lr"]["std"]
    data_range   = stats["hr_data_range"]       # 27.587
    hr_min_true  = stats["hr"]["min"]           # 9.06  (physical min)
    p1           = stats["hr_percentiles"]["p1"]
    p99          = stats["hr_percentiles"]["p99"]

    sep("STATS FROM stats.json")
    print(f"  hr_mean      = {hr_mean:.4f}")
    print(f"  hr_std       = {hr_std:.4f}")
    print(f"  lr_mean      = {lr_mean:.4f}")
    print(f"  lr_std       = {lr_std:.4f}")
    print(f"  DATA_RANGE   = {data_range:.4f}  (p99 - p1)")
    print(f"  p1           = {p1:.4f}")
    print(f"  p99          = {p99:.4f}")
    print(f"  hr_min_true  = {hr_min_true:.4f}")

    # ---- Dataset (1 batch only) ----
    sep("DATASET")
    USE_AUX = bool(args.use_aux)

    ds = SRDataset(
        split="train",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=USE_AUX,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "train" / "AUX") if USE_AUX else None,
        repeat_channels=False,
        transform=None,   # no augmentation for debug
    )

    print(f"  Train dataset size: {len(ds)} samples")

    loader = DataLoader(ds, batch_size=4, shuffle=False, num_workers=0)
    batch  = next(iter(loader))

    sep("RAW BATCH (normalised, as seen by model)")
    tensor_stats("lr",       batch["lr"])
    tensor_stats("hr",       batch["hr"])
    tensor_stats("hr_mask",  batch["hr_mask"])
    if "water_mask" in batch:
        tensor_stats("water_mask", batch["water_mask"])
    if "aux" in batch:
        tensor_stats("aux", batch["aux"])

    # ---- Mask coverage ----
    sep("MASK DIAGNOSTICS")
    hr_mask = batch["hr_mask"]
    print(f"  Valid pixel fraction (hr_mask): {hr_mask.float().mean():.4f}")

    # Per-sample bounding box check (mirrors _compute_single_metric_set)
    print("\n  Per-sample bounding box sizes:")
    for b in range(hr_mask.shape[0]):
        valid = hr_mask[b, 0] > 0.5
        if valid.sum() == 0:
            print(f"    sample {b}: NO VALID PIXELS")
            continue
        coords = torch.where(valid)
        ymin, ymax = coords[0].min().item(), coords[0].max().item()
        xmin, xmax = coords[1].min().item(), coords[1].max().item()
        h = ymax - ymin + 1
        w = xmax - xmin + 1
        flag = " <-- TOO SMALL FOR SSIM" if h < 11 or w < 11 else ""
        print(f"    sample {b}: H={h}  W={w}  "
              f"[y:{ymin}-{ymax}, x:{xmin}-{xmax}]{flag}")

    # ---- Water mask coverage ----
    if "water_mask" in batch:
        sep("WATER MASK DIAGNOSTICS")
        wm = batch["water_mask"]
        for b in range(wm.shape[0]):
            water_frac = (wm[b] * hr_mask[b]).float().mean().item()
            print(f"    sample {b}: water fraction = {water_frac:.4f}")

    # ---- AUX channels ----
    if USE_AUX and "aux" in batch:
        sep("AUX CHANNEL STATS (per channel)")
        aux = batch["aux"]
        aux_chans = aux.shape[1]
        print(f"  Total AUX channels: {aux_chans}")
        for c in range(aux_chans):
            ch = aux[:, c]
            print(f"    ch{c:02d}  min={ch.min():.4f}  max={ch.max():.4f}  "
                  f"mean={ch.mean():.4f}  nan={ch.isnan().sum().item()}")
    else:
        aux_chans = 0

    # ---- Model ----
    sep("MODEL FORWARD PASS")
    model = SwinIRModule(
        pretrained_path=None,           # skip pretrained for speed
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        data_range=data_range,
        hr_mean=hr_mean,
        hr_std=hr_std,
        adaptation_strategy=args.adaptation_strategy,
        aux_chans=aux_chans if USE_AUX else 1,  # at least 1 required
    )
    model.eval()
    print(f"  Strategy: {args.adaptation_strategy}")

    with torch.no_grad():
        sr_img = model(batch)

    sep("MODEL OUTPUT (normalised)")
    tensor_stats("sr_img (raw)", sr_img)

    # ---- Denormalise ----
    sr_denorm  = model.denormalize(sr_img[:, 0:1])
    hr_denorm  = model.denormalize(batch["hr"][:, 0:1])

    sep("DENORMALISED VALUES (should be in Celsius ~20-48)")
    tensor_stats("sr  (denorm)", sr_denorm)
    tensor_stats("hr  (denorm)", hr_denorm)
    print(f"\n  Expected physical range:  p1={p1:.2f}°C  p99={p99:.2f}°C")
    print(f"  SR within [p1, p99]?  "
          f"min_ok={sr_denorm.min().item() >= p1}  "
          f"max_ok={sr_denorm.max().item() <= p99}")

    # ---- SSIM input check ----
    sep("SSIM INPUT CHECK  (after normalization to [0,1])")
    sr_norm = (sr_denorm - p1) / (p99 - p1)
    hr_norm = (hr_denorm - p1) / (p99 - p1)
    sr_norm_c = sr_norm.clamp(0, 1)
    hr_norm_c = hr_norm.clamp(0, 1)

    tensor_stats("sr_norm (pre-clamp)",  sr_norm)
    tensor_stats("hr_norm (pre-clamp)",  hr_norm)
    tensor_stats("sr_norm (clamped)",    sr_norm_c)
    tensor_stats("hr_norm (clamped)",    hr_norm_c)

    clamp_frac = ((sr_norm < 0) | (sr_norm > 1)).float().mean().item()
    print(f"\n  Fraction of SR pixels clamped: {clamp_frac:.4f}  "
          f"({'OK' if clamp_frac < 0.05 else 'HIGH -- SR is out of range early in training'})")

    # ---- Quick SSIM sanity ----
    sep("SSIM SANITY (first valid sample)")
    try:
        from torchmetrics.image import StructuralSimilarityIndexMeasure
        ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0)

        for b in range(sr_norm_c.shape[0]):
            valid = hr_mask[b, 0] > 0.5
            if valid.sum() == 0:
                continue
            coords = torch.where(valid)
            ymin, ymax = coords[0].min().item(), coords[0].max().item()
            xmin, xmax = coords[1].min().item(), coords[1].max().item()
            h, w = ymax - ymin + 1, xmax - xmin + 1
            if h < 11 or w < 11:
                print(f"  sample {b}: crop too small ({h}x{w}), skipping")
                continue

            sr_crop = sr_norm_c[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
            hr_crop = hr_norm_c[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
            score = ssim_fn(sr_crop, hr_crop)
            print(f"  sample {b}: SSIM={score:.4f}  crop=({h}x{w})  "
                  f"{'OK' if score >= 0 else 'NEGATIVE -- problem remains'}")
            break

    except Exception as e:
        print(f"  SSIM check failed: {e}")

    sep("DONE")
    print("  If SR (denorm) range is wildly outside [20, 48], the network")
    print("  needs more training — negative SSIM is expected at epoch 1.")
    print("  If SR is in range but SSIM is still negative, the data_range")
    print("  passed to StructuralSimilarityIndexMeasure is wrong.")
    print()


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--adaptation_strategy", default="projection",
                   choices=["projection", "direct", "fusion"])
    p.add_argument("--use_aux", type=int, default=1)
    args = p.parse_args()
    main(args)