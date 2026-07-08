"""
spatial_leakage_diagnostics.py

Two independent diagnostics for the train/val spatial block split:

1. EMPIRICAL VARIOGRAM (data-only, no model needed)
   Estimates the spatial autocorrelation range of HR temperature along the
   row (north-south) axis -- the axis your blocks are split on. Lets you
   compare the autocorrelation range against block_size_pixels (1024 px,
   ~7680 m) to argue whether a zero-width boundary between adjacent
   train/val blocks is likely to leak meaningfully.

2. BOUNDARY-DISTANCE vs RESIDUAL CHECK (needs model predictions)
   For every val tile, computes the distance (in metres) from the tile to
   the nearest train/val block boundary within its own raster, then bins
   val error by that distance. If error near boundaries is not
   systematically lower than error in block interiors, that's direct
   empirical evidence the boundary leakage is small in your data.

Run diagnostic 1 any time. Run diagnostic 2 once you have per-tile val
residuals (e.g. saved as a JSON: {tile_name: mae_or_rmse, ...}).

Usage:
    python spatial_leakage_diagnostics.py variogram
    python spatial_leakage_diagnostics.py boundary --residuals val_residuals.json
"""

import json
import argparse
import numpy as np
import rasterio
import matplotlib.pyplot as plt
from pathlib import Path

BASE       = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR     = BASE / "HR_downsampled"
SPLIT_JSON = BASE / "split_map.json"
META_JSON  = BASE / "processed/patches/metadata.json"
OUT_DIR    = Path("/share/home/e2406751/Superresolution-TIR/results/diagnostics")
OUT_DIR.mkdir(parents=True, exist_ok=True)

PIXEL_SIZE_M = 7.5  # HR pixel resolution

TRAINVAL_RASTERS = [
    "BAS_2025.tif",
    "DZM_2013.tif", "DZM_2014.tif", "DZM_2019.tif", "DZM_2023.tif",
    "PDR_2013.tif", "PDR_2014.tif", "PDR_2023.tif",
]


# ---------- Diagnostic 1: empirical variogram along row axis ----------

def build_valid_mask(arr, nodata):
    valid = np.isfinite(arr)
    if nodata is None:
        return valid
    if np.isnan(nodata):
        return valid & ~np.isnan(arr)
    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return valid & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)


def semivariance_for_raster(hr_path, lags_px, n_samples_per_lag=20000, rng=None):
    """
    Sample random (row, col) locations; for each lag h (in pixels),
    compare z(row, col) to z(row+h, col) -- same column, row-shifted --
    since blocking is along the row axis. Returns mean semivariance per lag
    and the number of valid pairs actually used.
    """
    rng = rng or np.random.default_rng(0)
    with rasterio.open(hr_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata
    if nodata is not None:
        data[data < -1e30] = np.nan
    valid = build_valid_mask(data, nodata)

    n_rows, n_cols = data.shape
    results = {}
    for h in lags_px:
        if h >= n_rows:
            continue
        rows = rng.integers(0, n_rows - h, size=n_samples_per_lag)
        cols = rng.integers(0, n_cols, size=n_samples_per_lag)

        z1 = data[rows, cols]
        z2 = data[rows + h, cols]
        v1 = valid[rows, cols]
        v2 = valid[rows + h, cols]
        keep = v1 & v2
        if keep.sum() < 30:
            continue
        diffs = z1[keep] - z2[keep]
        gamma = 0.5 * np.mean(diffs ** 2)
        results[h] = {"gamma": float(gamma), "n_pairs": int(keep.sum())}
    return results


def run_variogram():
    lags_px = [1, 2, 4, 8, 16, 32, 64, 96, 128, 192, 256, 384, 512, 768, 1024, 1536]

    all_gammas = {h: [] for h in lags_px}
    per_raster = {}

    for fname in TRAINVAL_RASTERS:
        path = HR_DIR / fname
        if not path.exists():
            print(f"  skip {fname} (not found)")
            continue
        print(f"  computing variogram: {fname}")
        res = semivariance_for_raster(path, lags_px)
        per_raster[fname] = res
        for h, v in res.items():
            all_gammas[h].append(v["gamma"])

    lag_m = []
    gamma_mean = []
    for h in lags_px:
        if all_gammas[h]:
            lag_m.append(h * PIXEL_SIZE_M)
            gamma_mean.append(np.mean(all_gammas[h]))

    # crude "range" estimate: distance at which gamma reaches 95% of the
    # plateau (mean of the last 3 lags), i.e. where autocorrelation has
    # effectively died out
    sill_est = np.mean(gamma_mean[-3:])
    range_m = None
    for d, g in zip(lag_m, gamma_mean):
        if g >= 0.95 * sill_est:
            range_m = d
            break

    block_size_m = 1024 * PIXEL_SIZE_M  # matches split_strategy.py

    print("\n--- Variogram summary (pooled across train/val campaigns) ---")
    for d, g in zip(lag_m, gamma_mean):
        print(f"  lag {d:>7.1f} m   semivariance {g:.4f}")
    print(f"\nEstimated sill: {sill_est:.4f}")
    print(f"Estimated autocorrelation range: {range_m} m")
    print(f"Block size used in split_strategy.py: {block_size_m:.0f} m")
    if range_m is not None:
        ratio = block_size_m / range_m
        print(f"Block size is ~{ratio:.1f}x the estimated autocorrelation range.")
        print("Rule of thumb: if this ratio is well above 1, a zero-width")
        print("boundary between adjacent blocks affects only a thin sliver")
        print("of each block's area, and leakage is likely minor.")

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.plot(lag_m, gamma_mean, marker="o")
    ax.axhline(sill_est, color="gray", linestyle="--", label="estimated sill")
    if range_m is not None:
        ax.axvline(range_m, color="red", linestyle="--", label=f"estimated range ({range_m:.0f} m)")
    ax.axvline(block_size_m, color="green", linestyle=":", label=f"block size ({block_size_m:.0f} m)")
    ax.set_xlabel("Lag distance along row axis (m)")
    ax.set_ylabel("Semivariance (°C²)")
    ax.set_title("Empirical variogram, HR temperature, row-axis lags")
    ax.legend()
    fig.tight_layout()
    out_path = OUT_DIR / "variogram_row_axis.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    with open(OUT_DIR / "variogram_results.json", "w") as f:
        json.dump({
            "per_raster": per_raster,
            "pooled_lag_m": lag_m,
            "pooled_gamma": gamma_mean,
            "sill_estimate": sill_est,
            "range_estimate_m": range_m,
            "block_size_m": block_size_m,
        }, f, indent=2)


# ---------- Diagnostic 2: boundary distance vs residual ----------

def boundary_rows_for_raster(fname, split_map):
    """Row indices where the split changes (train<->val) within one raster."""
    blocks = sorted(split_map[fname]["blocks"], key=lambda b: b["row_start"])
    boundaries = []
    for i in range(len(blocks) - 1):
        if blocks[i]["split"] != blocks[i + 1]["split"]:
            boundaries.append(blocks[i]["row_end"])  # == blocks[i+1]["row_start"]
    return boundaries


def run_boundary_check(residuals_path):
    with open(SPLIT_JSON) as f:
        split_map = json.load(f)
    with open(META_JSON) as f:
        metadata = json.load(f)
    with open(residuals_path) as f:
        residuals = json.load(f)  # {tile_name: error_value}

    boundary_rows_cache = {
        fname: boundary_rows_for_raster(fname, split_map)
        for fname in split_map
    }

    records = []
    for tile_name, meta in metadata.items():
        if meta["split"] not in ("train", "val"):
            continue
        if tile_name not in residuals:
            continue
        fname = meta["hr_source"]
        boundaries = boundary_rows_cache.get(fname, [])
        if not boundaries:
            continue

        tile_center_row = meta["row_origin"] + 128  # HR_TILE // 2
        dist_px = min(abs(tile_center_row - b) for b in boundaries)
        dist_m = dist_px * PIXEL_SIZE_M

        records.append({
            "tile_name": tile_name,
            "split": meta["split"],
            "campaign": meta["campaign"],
            "dist_to_boundary_m": dist_m,
            "error": residuals[tile_name],
        })

    if not records:
        print("No matching tiles found -- check that residuals keys match")
        print("tile names in metadata.json, and that split_map.json has")
        print("mixed-split rasters.")
        return

    val_records = [r for r in records if r["split"] == "val"]
    if not val_records:
        print("No val tiles found in residuals -- nothing to check.")
        return

    dists = np.array([r["dist_to_boundary_m"] for r in val_records])
    errs = np.array([r["error"] for r in val_records])

    # bin by distance
    bin_edges = [0, 128 * PIXEL_SIZE_M, 256 * PIXEL_SIZE_M, 512 * PIXEL_SIZE_M, np.inf]
    bin_labels = ["0-960m", "960-1920m", "1920-3840m", ">3840m"]
    print("\n--- Val error by distance to nearest train/val boundary ---")
    for lo, hi, label in zip(bin_edges[:-1], bin_edges[1:], bin_labels):
        mask = (dists >= lo) & (dists < hi)
        if mask.sum() == 0:
            print(f"  {label:>12}: n=0")
            continue
        print(f"  {label:>12}: n={mask.sum():4d}  mean_error={errs[mask].mean():.4f}  "
              f"std={errs[mask].std():.4f}")

    corr = np.corrcoef(dists, errs)[0, 1]
    print(f"\nPearson correlation (distance vs error): {corr:.3f}")
    print("If this is close to 0 or slightly negative rather than clearly")
    print("positive (error decreasing with distance from boundary), there is")
    print("no evidence tiles near split boundaries are 'easier' due to leakage.")

    # plot
    fig, ax = plt.subplots(figsize=(7, 5))
    ax.scatter(dists, errs, alpha=0.4, s=12)
    ax.set_xlabel("Distance from val tile to nearest train/val boundary (m)")
    ax.set_ylabel("Val error (as provided)")
    ax.set_title(f"Boundary-distance vs error (r = {corr:.3f})")
    fig.tight_layout()
    out_path = OUT_DIR / "boundary_distance_vs_error.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nPlot saved to {out_path}")

    with open(OUT_DIR / "boundary_distance_results.json", "w") as f:
        json.dump(records, f, indent=2)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    sub.add_parser("variogram")
    b = sub.add_parser("boundary")
    b.add_argument("--residuals", required=True,
                    help="JSON file: {tile_name: error_value}")
    args = parser.parse_args()

    if args.mode == "variogram":
        run_variogram()
    else:
        run_boundary_check(args.residuals)