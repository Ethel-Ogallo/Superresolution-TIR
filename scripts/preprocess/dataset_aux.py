"""
Patch auxiliary rasters using EXACT LR tile grid.

FINAL AUX CHANNELS
------------------
0 BLUE
1 GREEN
2 RED
3 NIR
4 SWIR1
5 NDVI
6 NDWI
7 NDMI
8 LULC
9 DEM

INPUT AUX TIFF CHANNELS
-----------------------
0 B02 BLUE
1 B03 GREEN
2 B04 RED
3 B05 RED_EDGE      <- REMOVED
4 B08 NIR
5 B11 SWIR1
6 NDVI
7 NDWI
8 NDMI
9 LULC
10 DEM

This script:
- aligns AUX patches to LR tile grid
- removes RED_EDGE band
- normalizes spectral reflectance
- clamps spectral indices
- scales DEM
- preserves categorical LULC
- pads edge tiles safely
- saves training-ready AUX tensors
"""

import json
import numpy as np
import rasterio
from pathlib import Path
from rasterio.windows import Window

# PATHS
PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
METADATA_PATH = PATCHES_DIR / "metadata.json"

TILE_SIZE = 64

# FINAL AUX CHANNEL DEFINITIONS
AUX_CHANNELS = [
    "BLUE",
    "GREEN",
    "RED",
    "NIR",
    "SWIR1",
    "NDVI",
    "NDWI",
    "NDMI",
    "LULC",
    "DEM"
]

print("\nFINAL AUX CHANNELS")
for i, name in enumerate(AUX_CHANNELS):
    print(f"{i}: {name}")

# INPUT TIFF CHANNEL SELECTION
# REMOVE RED_EDGE BAND (old channel 3)
KEEP_CHANNELS = [0, 1, 2, 4, 5, 6, 7, 8, 9, 10]

# OUTPUT FOLDERS
for split in ["train", "val", "test"]:
    (PATCHES_DIR / split / "AUX").mkdir(
        parents=True,
        exist_ok=True
    )

# LOAD TILE METADATA
with open(METADATA_PATH) as f:
    metadata = json.load(f)

print(f"\nProcessing {len(metadata)} AUX patches...\n")

# NORMALIZATION FUNCTIONS
def normalize_reflectance(x):
    """
    Normalize Sentinel/Landsat reflectance.
    Approx:
        0–10000 -> 0–1
    """
    x = x / 10000.0
    # prevent rare outliers
    x = np.clip(x, 0.0, 1.5)
    return x

def normalize_index(x):
    """NDVI / NDWI / NDMI"""
    return np.clip(x, -1.0, 1.0)


def normalize_dem(x):
    """DEM scaling."""
    return x / 2000.0


def normalize_stack(arr):
    """
    Input:
        (10, H, W)

    Output:
        normalized float32 tensor
    """

    out = np.zeros_like(arr, dtype=np.float32)

    # RAW SPECTRAL BANDS BLUE GREEN RED NIR SWIR1
    out[0:5] = normalize_reflectance(arr[0:5])

    # INDICES
    out[5] = normalize_index(arr[5])  # NDVI
    out[6] = normalize_index(arr[6])  # NDWI
    out[7] = normalize_index(arr[7])  # NDMI

    # LULC (categorical)
    out[8] = arr[8]

    # DEM
    out[9] = normalize_dem(arr[9])

    return out.astype(np.float32)

# CACHE OPEN RASTERS
open_aux = {}

saved = {
    "train": 0,
    "val": 0,
    "test": 0
}

missing = set()

# MAIN PATCHING LOOP
for tile_name, info in metadata.items():

    campaign = info["campaign"]
    split = info["split"]

    row = info["row_origin"]
    col = info["col_origin"]

    aux_path = AUX_DIR / f"{campaign}_aux.tif"

    out_path = (
        PATCHES_DIR
        / split
        / "AUX"
        / f"{tile_name}.npy"
    )

    # HANDLE MISSING AUX
    if not aux_path.exists():

        if campaign not in missing:
            print(f"[WARNING] Missing AUX: {campaign}")
            missing.add(campaign)

        zero_patch = np.zeros(
            (10, TILE_SIZE, TILE_SIZE),
            dtype=np.float32
        )

        np.save(out_path, zero_patch)

        saved[split] += 1
        continue

    # OPEN RASTER (CACHE)
    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(aux_path)

    src = open_aux[campaign]

    # SAFETY CHECK
    if src.count != 11:
        raise ValueError(
            f"{campaign} expected 11 bands "
            f"but found {src.count}"
        )

    # READ SAME WINDOW AS LR TILE
    window = Window(
        col_off=col,
        row_off=row,
        width=TILE_SIZE,
        height=TILE_SIZE
    )

    patch = src.read(window=window).astype(np.float32)

    # REMOVE RED_EDGE CHANNEL
    patch = patch[KEEP_CHANNELS]

    # EDGE TILE PADDING
    if patch.shape[1:] != (TILE_SIZE, TILE_SIZE):

        padded = np.zeros(
            (10, TILE_SIZE, TILE_SIZE),
            dtype=np.float32
        )

        h, w = patch.shape[1:]

        padded[:, :h, :w] = patch

        patch = padded

    # FINAL CHANNEL CHECK
    if patch.shape[0] != 10:
        raise ValueError(
            f"{campaign} produced "
            f"{patch.shape[0]} channels instead of 10"
        )

    # NORMALIZATION
    patch = normalize_stack(patch)

    # NAN / INF SAFETY
    patch = np.nan_to_num(
        patch,
        nan=0.0,
        posinf=0.0,
        neginf=0.0
    )

    # SAVE
    np.save(out_path, patch)

    saved[split] += 1

# CLEANUP
for src in open_aux.values():
    src.close()

# SUMMARY
print("\nDONE\n")

print(f"Train AUX patches: {saved['train']}")
print(f"Val AUX patches:   {saved['val']}")
print(f"Test AUX patches:  {saved['test']}")