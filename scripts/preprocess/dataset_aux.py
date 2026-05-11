# ----------------------------
# AUX PATCHING (FIXED VERSION)
# ----------------------------

import json
import numpy as np
import rasterio
from pathlib import Path
from rasterio.windows import Window

PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
METADATA_PATH = PATCHES_DIR / "metadata.json"

TILE_SIZE = 64


# ----------------------------
# ESA WORLD COVER CLASSES
# ----------------------------
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
LULC_MAP = {v: i for i, v in enumerate(LULC_CLASSES)}


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
# STACK BUILDER (FIXED)
# ----------------------------
def build_aux_stack(arr):
    """
    EXPECTED INPUT ORDER (10 bands):
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
    """

    # spectral (5)
    spectral = normalize_reflectance(arr[0:5])

    # indices (3)
    ndvi = normalize_index(arr[5])[None]
    ndwi = normalize_index(arr[6])[None]
    ndmi = normalize_index(arr[7])[None]

    # categorical → one-hot (11)
    lulc = encode_lulc_onehot(arr[8])

    # continuous
    dem = normalize_dem(arr[9])[None]

    return np.concatenate(
        [spectral, ndvi, ndwi, ndmi, lulc, dem],
        axis=0
    ).astype(np.float32)


# ----------------------------
# LOAD METADATA
# ----------------------------
with open(METADATA_PATH) as f:
    metadata = json.load(f)


open_aux = {}
saved = {"train": 0, "val": 0, "test": 0}
missing = set()


# ----------------------------
# MAIN LOOP
# ----------------------------
for tile_name, info in metadata.items():

    campaign = info["campaign"]
    split = info["split"]
    row = info["row_origin"]
    col = info["col_origin"]

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

        # 20 channels = 5 + 3 + 11 + 1
        np.save(out_path, np.zeros((20, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # OPEN RASTER
    # ----------------------------
    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(aux_path)

    src = open_aux[campaign]

    window = Window(col, row, TILE_SIZE, TILE_SIZE)
    patch = src.read(window=window).astype(np.float32)

    # REMOVE RED EDGE IF PRESENT (safety)
    if patch.shape[0] == 11:
        patch = np.delete(patch, 3, axis=0)

    # PAD IF NECESSARY
    if patch.shape[1:] != (TILE_SIZE, TILE_SIZE):
        padded = np.zeros((10, TILE_SIZE, TILE_SIZE), np.float32)
        h, w = patch.shape[1:]
        padded[:, :h, :w] = patch
        patch = padded

    # ----------------------------
    # BUILD FINAL AUX STACK
    # ----------------------------
    patch = build_aux_stack(patch)

    # CLEAN
    patch = np.nan_to_num(patch, nan=0.0, posinf=0.0, neginf=0.0)

    np.save(out_path, patch)
    saved[split] += 1


# ----------------------------
# CLEANUP
# ----------------------------
for src in open_aux.values():
    src.close()

print("\nDONE\n")
print(saved)