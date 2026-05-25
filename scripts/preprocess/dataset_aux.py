# ----------------------------
# CLEAN AUX PATCHING SCRIPT (FINAL)
# ----------------------------

import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from pyproj import Transformer
from pathlib import Path

# ----------------------------
# PATHS
# ----------------------------
PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR     = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
METADATA_PATH = PATCHES_DIR / "metadata.json"
STATS_PATH = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/dem_stats.json")

TILE_SIZE = 64

# ----------------------------
# LOAD DEM STATS
# ----------------------------
with open(STATS_PATH) as f:
    dem_stats = json.load(f)

# ----------------------------
# TRANSFORMER
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
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]

def encode_lulc_onehot(x):
    h, w = x.shape
    out = np.zeros((len(LULC_CLASSES), h, w), dtype=np.float32)

    for i, cls in enumerate(LULC_CLASSES):
        out[i] = np.isclose(x, cls).astype(np.float32)

    return out

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
    bounds = info["bounds_2154"]

    aux_path = AUX_DIR / f"{campaign}_aux.tif"
    out_path = PATCHES_DIR / split / "AUX" / f"{tile_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    # ----------------------------
    # MISSING AUX HANDLING
    # ----------------------------
    if not aux_path.exists():
        if campaign not in missing:
            print(f"[WARNING] Missing AUX: {campaign}")
            missing.add(campaign)

        np.save(out_path, np.zeros((21, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # OPEN RASTER
    # ----------------------------
    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(aux_path)

    src = open_aux[campaign]

    # ----------------------------
    # WINDOW
    # ----------------------------
    west, south = transformer.transform(bounds["left"], bounds["bottom"])
    east, north = transformer.transform(bounds["right"], bounds["top"])

    window = from_bounds(west, south, east, north, src.transform)

    # ----------------------------
    # READ DATA (FIXED SCHEMA)
    # ----------------------------
    try:
        spectral = src.read(
            indexes=[1, 2, 3, 4, 5],
            window=window,
            out_shape=(5, TILE_SIZE, TILE_SIZE),
            resampling=Resampling.bilinear
        ).astype(np.float32)

        ndvi = src.read(6, window=window,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        resampling=Resampling.bilinear).astype(np.float32)

        ndwi = src.read(7, window=window,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        resampling=Resampling.bilinear).astype(np.float32)

        ndmi = src.read(8, window=window,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        resampling=Resampling.bilinear).astype(np.float32)

        lulc = src.read(9, window=window,
                        out_shape=(TILE_SIZE, TILE_SIZE),
                        resampling=Resampling.nearest).astype(np.float32)

        dem = src.read(10, window=window,
                    out_shape=(TILE_SIZE, TILE_SIZE),
                    resampling=Resampling.bilinear).astype(np.float32)

    except Exception as e:
        print(f"[WARNING] Failed {tile_name}: {e}")
        np.save(out_path, np.zeros((19, TILE_SIZE, TILE_SIZE), np.float32))
        saved[split] += 1
        continue

    # ----------------------------
    # NORMALIZATION
    # ----------------------------
    stats = dem_stats[campaign]

    spectral = normalize_spectral(spectral)
    ndvi = normalize_index(ndvi)
    ndwi = normalize_index(ndwi)
    ndmi = normalize_index(ndmi)
    dem = normalize_dem(dem, stats["p2"], stats["p98"])

    # ----------------------------
    # LULC FIX + ENCODING
    # ----------------------------
    fixed = np.zeros_like(lulc)
    for cls in LULC_CLASSES:
        fixed[np.isclose(lulc, cls)] = cls
    lulc = fixed

    lulc_onehot = encode_lulc_onehot(lulc)

    # ----------------------------
    # FINAL STACK
    # ----------------------------
    patch = np.concatenate(
        [
            spectral,        # 5
            ndvi[None],      # 1
            ndwi[None],      # 1
            ndmi[None],      # 1
            lulc_onehot,     # 11
            dem[None]        # 1
        ],
        axis=0
    )

    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

    np.save(out_path, patch.astype(np.float32))
    saved[split] += 1

# ----------------------------
# CLEANUP
# ----------------------------
for src in open_aux.values():
    src.close()

print(f"\nDONE, saved AUX patches: {sum(saved.values())}")


# | Block        | Channels |
# | ------------ | -------- |
# | Spectral     | 5        |
# | NDVI         | 1        |
# | NDWI         | 1        |
# | NDMI         | 1        |
# | LULC one-hot | 11       |
# | DEM          | 1        |
