# scripts/preprocess/tile_water_masks.py
"""
Tile water masks using existing patch metadata.
Reads row/col origins from metadata.json and extracts
256x256 windows from full water mask rasters.
Saves to patches/{split}/WM/ alongside HR and LR.
"""

import json
import numpy as np
import rasterio
from pathlib import Path

PATCHES_DIR    = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches2")
WATER_MASK_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/water_masks")
METADATA_PATH  = PATCHES_DIR / "metadata.json"
TILE_SIZE      = 256

# Create WM folders
for split in ["train", "val", "test"]:
    (PATCHES_DIR / split / "WM").mkdir(parents=True, exist_ok=True)

# Load metadata
with open(METADATA_PATH) as f:
    metadata = json.load(f)

print(f"Processing {len(metadata)} tiles...")

# Cache open rasters per campaign
open_masks = {}
saved = {"train": 0, "val": 0, "test": 0}
missing = set()

for tile_name, info in metadata.items():
    campaign = info["campaign"]          # e.g. BRC_2022
    split    = info["split"]            # train/val/test
    row      = info["row_origin"]
    col      = info["col_origin"]

    wm_path  = WATER_MASK_DIR / f"{campaign}_water_mask.tif"
    out_path = PATCHES_DIR / split / "WM" / f"{tile_name}.npy"

    if not wm_path.exists():
        if campaign not in missing:
            print(f"  [WARNING] No water mask for {campaign} — saving zeros")
            missing.add(campaign)
        np.save(out_path, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32))
        saved[split] += 1
        continue

    # Cache raster open
    if campaign not in open_masks:
        open_masks[campaign] = rasterio.open(wm_path)
    src = open_masks[campaign]

    window  = rasterio.windows.Window(
        col_off = col,
        row_off = row,
        width   = TILE_SIZE,
        height  = TILE_SIZE,
    )
    wm_tile = src.read(1, window=window).astype(np.float32)

    # Pad edge tiles
    if wm_tile.shape != (TILE_SIZE, TILE_SIZE):
        padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        padded[:wm_tile.shape[0], :wm_tile.shape[1]] = wm_tile
        wm_tile = padded

    np.save(out_path, wm_tile)
    saved[split] += 1

# Close all open rasters
for src in open_masks.values():
    src.close()

print(f"\nDone.")
print(f"  Train WM tiles: {saved['train']}")
print(f"  Val   WM tiles: {saved['val']}")
print(f"  Test  WM tiles: {saved['test']}")

# Quick sanity check — water pixel stats per split
print(f"\nWater pixel stats:")
for split in ["train", "val", "test"]:
    wm_dir = PATCHES_DIR / split / "WM"
    files  = sorted(wm_dir.glob("*.npy"))
    if not files:
        continue
    water_fracs = [np.load(f).mean() for f in files]
    print(f"  {split}: mean water frac = {np.mean(water_fracs):.4f} "
          f"(min={np.min(water_fracs):.4f}, max={np.max(water_fracs):.4f})")


