# ----------------------------
# AUX SPADE PATCHING SCRIPT
# Generates:
#   AUX_SPADE_HR  -> (20,256,256)
#   AUX_SPADE_MID -> (20,128,128)
# ----------------------------

import json
import numpy as np
import rasterio
import cv2
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
BASE_DIR = Path("/share/home/e2406751/Superresolution-TIR")

PATCHES_DIR = BASE_DIR / "data/processed/patches"
AUX_DIR = BASE_DIR / "data/AUX/auxiliary"

METADATA_PATH = PATCHES_DIR / "metadata.json"
STATS_PATH = BASE_DIR / "data/AUX/dem_stats.json"

# ----------------------------
# SIZES
# ----------------------------
HR_SIZE = 256
MID_SIZE = 128

# ----------------------------
# LOAD DEM STATS
# ----------------------------
with open(STATS_PATH) as f:
    dem_stats = json.load(f)

# ----------------------------
# COORD TRANSFORM
# ----------------------------
transformer = Transformer.from_crs(
    "EPSG:2154",
    "EPSG:4326",
    always_xy=True
)

# ----------------------------
# NORMALIZATION
# ----------------------------
def normalize_spectral(x):
    return np.clip(x / 10000.0, 0.0, 1.0)

def normalize_index(x):
    return np.clip(x, -1.0, 1.0)

def normalize_dem(x, p2, p98):
    x = np.clip(x, p2, p98)
    return (x - p2) / (p98 - p2 + 1e-6)

# ----------------------------
# LULC ENCODING
# ----------------------------
LULC_CLASSES = [10,20,30,40,50,60,70,80,90,95,100]

def encode_lulc_onehot(x):

    h, w = x.shape

    out = np.zeros(
        (len(LULC_CLASSES), h, w),
        dtype=np.float32
    )

    for i, cls in enumerate(LULC_CLASSES):
        out[i] = np.isclose(x, cls).astype(np.float32)

    return out


# ----------------------------
# SAFE RESIZE
# Preserves categorical channels
# ----------------------------
def resize_patch(patch, size):

    c, _, _ = patch.shape

    out = np.zeros(
        (c, size, size),
        dtype=np.float32
    )

    for i in range(c):

        # LULC one-hot channels:
        # channel indices 8-18
        # [5 spectral +3 indices = first 8]
        if 8 <= i <= 18:
            interp = cv2.INTER_NEAREST
        else:
            interp = cv2.INTER_LINEAR

        out[i] = cv2.resize(
            patch[i],
            (size, size),
            interpolation=interp
        )

    return out


# ----------------------------
# LOAD METADATA
# ----------------------------
with open(METADATA_PATH) as f:
    metadata = json.load(f)

open_aux = {}

saved = {
    "train":0,
    "val":0,
    "test":0
}

missing = set()

# ----------------------------
# MAIN LOOP
# ----------------------------
for tile_name, info in metadata.items():

    campaign = info["campaign"]
    split = info["split"]
    bounds = info["bounds_2154"]

    aux_path = AUX_DIR / f"{campaign}_aux.tif"

    mid_path = (
        PATCHES_DIR /
        split /
        "AUX_SPADE_MID" /
        f"{tile_name}.npy"
    )

    hr_path = (
        PATCHES_DIR /
        split /
        "AUX_SPADE_HR" /
        f"{tile_name}.npy"
    )

    mid_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    hr_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # ----------------------------
    # MISSING AUX
    # ----------------------------
    if not aux_path.exists():

        if campaign not in missing:
            print(
                f"[WARNING] Missing AUX: {campaign}"
            )
            missing.add(campaign)

        np.save(
            mid_path,
            np.zeros(
                (20, MID_SIZE, MID_SIZE),
                dtype=np.float32
            )
        )

        np.save(
            hr_path,
            np.zeros(
                (20, HR_SIZE, HR_SIZE),
                dtype=np.float32
            )
        )

        continue


    # ----------------------------
    # OPEN ONCE
    # ----------------------------
    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(
            aux_path
        )

    src = open_aux[campaign]

    # ----------------------------
    # WINDOW
    # ----------------------------
    west, south = transformer.transform(
        bounds["left"],
        bounds["bottom"]
    )

    east, north = transformer.transform(
        bounds["right"],
        bounds["top"]
    )

    window = from_bounds(
        west,
        south,
        east,
        north,
        src.transform
    )

    try:

        # ----------------------------
        # READ AT HR GRID
        # ----------------------------

        spectral = src.read(
            indexes=[1,2,3,4,5],
            window=window,
            out_shape=(5,HR_SIZE,HR_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        ndvi = src.read(
            6,
            window=window,
            out_shape=(HR_SIZE,HR_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        ndwi = src.read(
            7,
            window=window,
            out_shape=(HR_SIZE,HR_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        ndmi = src.read(
            8,
            window=window,
            out_shape=(HR_SIZE,HR_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        lulc = src.read(
            9,
            window=window,
            out_shape=(HR_SIZE,HR_SIZE),
            resampling=Resampling.nearest
        ).astype(np.float32)

        dem = src.read(
            10,
            window=window,
            out_shape=(HR_SIZE,HR_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)


    except Exception as e:

        print(
            f"[WARNING] Failed {tile_name}: {e}"
        )

        continue


    # ----------------------------
    # NORMALIZATION
    # ----------------------------
    stats = dem_stats[campaign]

    spectral = normalize_spectral(
        spectral
    )

    ndvi = normalize_index(ndvi)
    ndwi = normalize_index(ndwi)
    ndmi = normalize_index(ndmi)

    dem = normalize_dem(
        dem,
        stats["p2"],
        stats["p98"]
    )

    # ----------------------------
    # LULC FIX
    # ----------------------------
    fixed = np.zeros_like(lulc)

    for cls in LULC_CLASSES:
        fixed[
            np.isclose(
                lulc,
                cls
            )
        ] = cls

    lulc_onehot = encode_lulc_onehot(
        fixed
    )

    # ----------------------------
    # STACK HR
    # ----------------------------
    aux_hr = np.concatenate(
        [
            spectral,
            ndvi[None],
            ndwi[None],
            ndmi[None],
            lulc_onehot,
            dem[None]
        ],
        axis=0
    )

    aux_hr = np.nan_to_num(
        aux_hr,
        nan=0.0,
        posinf=1.0,
        neginf=0.0
    )

    # ----------------------------
    # MID FOR SPADE
    # ----------------------------
    aux_mid = resize_patch(
        aux_hr,
        MID_SIZE
    )

    # ----------------------------
    # SAVE
    # ----------------------------
    np.save(
        hr_path,
        aux_hr.astype(np.float32)
    )

    np.save(
        mid_path,
        aux_mid.astype(np.float32)
    )

    saved[split] += 1


# ----------------------------
# CLEANUP
# ----------------------------
for src in open_aux.values():
    src.close()

print(
    f"\nDONE: saved {sum(saved.values())} SPADE patches"
)