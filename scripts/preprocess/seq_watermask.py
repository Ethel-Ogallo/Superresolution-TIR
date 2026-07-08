# scripts/preprocess/tile_water_masks_vsr.py
"""
Tile water masks using the existing VSR patch metadata.
Reads row/col origins from processed_vsr/metadata.json, extracts
256x256 windows from full water mask rasters, and saves them
in the sequential Resolution-First framework: processed_vsr/{split}/WM/{seq_name}/00xx.npy
"""

import json
import numpy as np
import rasterio
from pathlib import Path

VSR_DIR        = Path("/share/home/e2406751/Superresolution-TIR/data/processed/seq_patches")
WATER_MASK_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/water_masks")
METADATA_PATH  = VSR_DIR / "metadata.json"
TILE_SIZE      = 256

# Load VSR metadata
if not METADATA_PATH.exists():
    raise FileNotFoundError(f"VSR metadata file not found at: {METADATA_PATH}")

with open(METADATA_PATH) as f:
    metadata = json.load(f)

print(f"Processing water masks for {len(metadata)} total frames...")

# Cache open rasters per campaign
open_masks = {}
saved = {"train": 0, "val": 0, "test": 0}
missing = set()

for frame_id, info in metadata.items():
    campaign = info["campaign"]          # e.g. BRC_2022
    split    = info["split"]             # train/val/test
    seq_name = info["sequence_group"]    # e.g. BRC_2022_b0_r1024_c384
    frame_f  = info["frame_index"]       # e.g. 0, 1, 2, 3, 4
    row      = info["row_origin"]
    col      = info["col_origin"]

    wm_path  = WATER_MASK_DIR / f"{campaign}_water_mask.tif"
    
    # Establish Resolution-First output location: Split / WM / Sequence / Frame
    out_dir  = VSR_DIR / split / "WM" / seq_name
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{frame_f:04d}.npy"

    # Missing mask fallback handler
    if not wm_path.exists():
        if campaign not in missing:
            print(f"  [WARNING] No water mask found for {campaign} — saving zeros")
            missing.add(campaign)
        np.save(out_path, np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32))
        saved[split] += 1
        continue

    # Raster handle caching strategy
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

    # Pad edge frames cutting through flight strip borders
    if wm_tile.shape != (TILE_SIZE, TILE_SIZE):
        padded = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.float32)
        padded[:wm_tile.shape[0], :wm_tile.shape[1]] = wm_tile
        wm_tile = padded

    np.save(out_path, wm_tile)
    saved[split] += 1

# Safely close raster handles
for src in open_masks.values():
    src.close()

print(f"\nDone.")
print(f"  Train WM frames saved: {saved['train']}")
print(f"  Val   WM frames saved: {saved['val']}")
print(f"  Test  WM frames saved: {saved['test']}")

# VSR Sequence Integrity Check — Evaluates actual water coverage statistics
print(f"\nWater pixel sequence stats:")
for split in ["train", "val", "test"]:
    split_wm_dir = VSR_DIR / split / "WM"
    if not split_wm_dir.exists():
        continue
        
    all_files = sorted(list(split_wm_dir.glob("**/*.npy")))
    if not all_files:
        continue
        
    water_fracs = [np.load(f).mean() for f in all_files]
    print(f"  {split}: mean water frac = {np.mean(water_fracs):.4f} "
          f"(min={np.min(water_fracs):.4f}, max={np.max(water_fracs):.4f})")