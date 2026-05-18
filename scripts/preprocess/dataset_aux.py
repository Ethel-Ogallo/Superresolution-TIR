# ----------------------------
# AUX PATCHING 
# ----------------------------

import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from pathlib import Path

PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR     = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
METADATA_PATH = PATCHES_DIR / "metadata.json"
TILE_SIZE = 64

# ----------------------------
# ESA WORLD COVER CLASSES
# ----------------------------
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]  # 11 classes
LULC_MAP = {v: i for i, v in enumerate(LULC_CLASSES)}

# ----------------------------
# COORDINATE TRANSFORMER
# EPSG:2154 → EPSG:4326
# ----------------------------
transformer = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_reflectance(x):
    return np.clip(x / 10000.0, 0.0, 1.5)

def normalize_index(x):
    return np.clip(x, -1.0, 1.0)

def normalize_dem(x):
    return x / 2000.0

# ----------------------------
# LULC ENCODING (ONE-HOT)
# ----------------------------
def encode_lulc_onehot(x):
    h, w = x.shape
    out = np.zeros((len(LULC_CLASSES), h, w), dtype=np.float32)
    for cls, idx in LULC_MAP.items():
        out[idx] = (x == cls).astype(np.float32)
    return out

# ----------------------------
# AUX STACK BUILDER 
# ----------------------------
def build_aux_stack(arr):
    """
    Input: 10 raw bands
    Output: 20 normalized channels
    """
    # Spectral (5 channels)
    spectral = np.clip(arr[0:5] / 10000.0, 0.0, 1.0)

    # Indices (3 channels)
    ndvi = np.clip(arr[5], -1.0, 1.0)[None, ...]
    ndwi = np.clip(arr[6], -1.0, 1.0)[None, ...]
    ndmi = np.clip(arr[7], -1.0, 1.0)[None, ...]

    # LULC One-hot (11 channels)
    lulc_raw = arr[8].astype(int)
    lulc_onehot = np.zeros((11, *lulc_raw.shape), dtype=np.float32)
    for i, cls in enumerate(LULC_CLASSES):
        lulc_onehot[i] = (lulc_raw == cls).astype(np.float32)

    # DEM (1 channel)
    dem = np.clip(arr[9] / 2000.0, 0.0, 1.5)[None, ...]

    stack = np.concatenate([spectral, ndvi, ndwi, ndmi, lulc_onehot, dem], axis=0)
    
    # Final safety
    stack = np.nan_to_num(stack, nan=0.0, posinf=1.0, neginf=-1.0)
    
    return stack.astype(np.float32)

# ----------------------------
# LOAD METADATA
# ----------------------------
with open(METADATA_PATH) as f:
    metadata = json.load(f)

open_aux = {}
saved   = {"train": 0, "val": 0, "test": 0}
missing = set()

# ----------------------------
# MAIN LOOP
# ----------------------------
for tile_name, info in metadata.items():

    campaign = info["campaign"]
    split    = info["split"]
    bounds   = info["bounds_2154"]   # ← USE GEOGRAPHIC BOUNDS

    aux_path = AUX_DIR / f"{campaign}_aux.tif"
    out_path = PATCHES_DIR / split / "AUX" / f"{tile_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # MISSING FILE HANDLING
    # ----------------------------
    if not aux_path.exists():
        if campaign not in missing:
            print(f"[WARNING] Missing AUX: {campaign}")
            missing.add(campaign)
        np.save(out_path, np.zeros((20, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # OPEN RASTER
    # ----------------------------
    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(aux_path)
    src = open_aux[campaign]

    # ----------------------------
    # CONVERT BOUNDS FROM EPSG:2154 → AUX CRS
    # ----------------------------
    # transform patch corners to aux raster CRS (EPSG:4326)
    west,  south = transformer.transform(bounds["left"],  bounds["bottom"])
    east,  north = transformer.transform(bounds["right"], bounds["top"])

    # get window in aux raster pixel coordinates
    window = from_bounds(west, south, east, north, src.transform)

    # ----------------------------
    # READ AND RESAMPLE TO TILE_SIZE
    # ----------------------------
    try:
        patch = src.read(
            window=window,
            out_shape=(src.count, TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

    except Exception as e:
        print(f"[WARNING] Failed to read {tile_name}: {e}")
        np.save(out_path, np.zeros((20, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # SAFETY: check band count
    # ----------------------------
    if patch.shape[0] == 11:
        patch = np.delete(patch, 3, axis=0)  # remove red edge if present

    if patch.shape[0] != 10:
        print(f"[WARNING] Unexpected band count {patch.shape[0]} for {tile_name}")
        np.save(out_path, np.zeros((20, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # BUILD FINAL AUX STACK
    # ----------------------------
    patch = build_aux_stack(patch)
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(out_path, patch)
    saved[split] += 1

# ----------------------------
# CLEANUP
# ----------------------------
for src in open_aux.values():
    src.close()

print("\nDONE")
print(saved)