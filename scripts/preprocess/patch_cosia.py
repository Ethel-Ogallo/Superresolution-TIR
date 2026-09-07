

import json
import numpy as np
import rasterio
from rasterio.windows import from_bounds
from rasterio.enums import Resampling
from rasterio.warp import transform_bounds
from pyproj import Transformer
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────

PATCHES_DIR   = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR       = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
COSIA_LC_DIR  = Path("/share/home/e2406751/Superresolution-TIR/data/processed/landcover")
METADATA_PATH = PATCHES_DIR / "metadata.json"
STATS_PATH    = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/dem_stats.json")

TILE_SIZE = 64

# ─── COSIA class definition ───────────────────────────────────────────────────

# Valid COSIA numero values (class 7 and 11 do not exist in the dataset)
COSIA_CLASSES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 0]
# 0 = unclassified/nodata — kept as a channel so the network knows coverage gaps
N_COSIA = len(COSIA_CLASSES)  # 14

# ─── Normalization ────────────────────────────────────────────────────────────

def normalize_spectral(x):
    return np.clip(x / 10000.0, 0.0, 1.0)

def normalize_index(x):
    return np.clip(x, -1.0, 1.0)

def normalize_dem(x, p2, p98):
    x = np.clip(x, p2, p98)
    return (x - p2) / (p98 - p2 + 1e-6)

# ─── COSIA one-hot ────────────────────────────────────────────────────────────

def encode_cosia_onehot(x: np.ndarray) -> np.ndarray:
    """
    x: (H, W) uint8 array of COSIA class ids (0 = nodata)
    returns: (N_COSIA, H, W) float32 one-hot array
    """
    h, w = x.shape
    out = np.zeros((N_COSIA, h, w), dtype=np.float32)
    for i, cls in enumerate(COSIA_CLASSES):
        out[i] = (x == cls).astype(np.float32)
    return out

# ─── Helpers ──────────────────────────────────────────────────────────────────

def read_window_from_4326_raster(src, west_4326, south_4326, east_4326, north_4326,
                                  bands, tile_size, resampling=Resampling.bilinear):
    """Read a spatial window from a raster in EPSG:4326."""
    window = from_bounds(west_4326, south_4326, east_4326, north_4326, src.transform)
    if isinstance(bands, int):
        return src.read(
            bands, window=window,
            out_shape=(tile_size, tile_size),
            resampling=resampling
        ).astype(np.float32)
    else:
        return src.read(
            indexes=bands, window=window,
            out_shape=(len(bands), tile_size, tile_size),
            resampling=resampling
        ).astype(np.float32)


def read_window_from_2154_raster(src, west_2154, south_2154, east_2154, north_2154,
                                  tile_size, resampling=Resampling.nearest):
    """Read a spatial window from a raster in EPSG:2154."""
    window = from_bounds(west_2154, south_2154, east_2154, north_2154, src.transform)
    return src.read(
        1, window=window,
        out_shape=(tile_size, tile_size),
        resampling=resampling
    )  # uint8, keep as-is for one-hot encoding

# ─── Load metadata & stats ────────────────────────────────────────────────────

with open(METADATA_PATH) as f:
    metadata = json.load(f)

with open(STATS_PATH) as f:
    dem_stats = json.load(f)

# ─── Coordinate transformer: EPSG:2154 → EPSG:4326 ───────────────────────────
# Used to convert patch bounds (stored in 2154) → 4326 for the aux.tif window.

to_4326 = Transformer.from_crs("EPSG:2154", "EPSG:4326", always_xy=True)

# ─── Open rasters once, close at the end ──────────────────────────────────────

open_aux   = {}   # campaign → rasterio src (EPSG:4326, 10 bands)
open_cosia = {}   # campaign → rasterio src (EPSG:2154, 1 band uint8)

saved   = {"train": 0, "val": 0, "test": 0}
missing = {"aux": set(), "cosia": set()}

# ─── Main loop ────────────────────────────────────────────────────────────────

for tile_name, info in metadata.items():

    campaign = info["campaign"]
    split    = info["split"]
    bounds   = info["bounds_2154"]   # keys: left, bottom, right, top  in EPSG:2154

    aux_path   = AUX_DIR     / f"{campaign}_aux.tif"
    cosia_path = COSIA_LC_DIR / f"{campaign}_landcover.tif"
    out_path   = PATCHES_DIR / split / "AUX" / f"{tile_name}.npy"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    TOTAL_CHANNELS = 5 + 1 + 1 + 1 + N_COSIA + 1  # = 23
    zeros = np.zeros((TOTAL_CHANNELS, TILE_SIZE, TILE_SIZE), np.float32)

    # ── Bounds in both CRS ───────────────────────────────────────────────────
    left_2154   = bounds["left"]
    bottom_2154 = bounds["bottom"]
    right_2154  = bounds["right"]
    top_2154    = bounds["top"]

    west_4326, south_4326 = to_4326.transform(left_2154,  bottom_2154)
    east_4326, north_4326 = to_4326.transform(right_2154, top_2154)

    # ── Open aux raster ──────────────────────────────────────────────────────
    if not aux_path.exists():
        if campaign not in missing["aux"]:
            print(f"[WARNING] Missing aux.tif: {campaign}")
            missing["aux"].add(campaign)
        np.save(out_path, zeros)
        saved[split] += 1
        continue

    if campaign not in open_aux:
        open_aux[campaign] = rasterio.open(aux_path)
    aux_src = open_aux[campaign]

    # ── Open COSIA landcover raster ──────────────────────────────────────────
    if not cosia_path.exists():
        if campaign not in missing["cosia"]:
            print(f"[WARNING] Missing COSIA landcover: {campaign} — LULC channels will be zero")
            missing["cosia"].add(campaign)
        cosia_src = None
    else:
        if campaign not in open_cosia:
            open_cosia[campaign] = rasterio.open(cosia_path)
        cosia_src = open_cosia[campaign]

    # ── Read aux bands ───────────────────────────────────────────────────────
    try:
        spectral = read_window_from_4326_raster(
            aux_src, west_4326, south_4326, east_4326, north_4326,
            bands=[1, 2, 3, 4, 5], tile_size=TILE_SIZE
        )
        ndvi = read_window_from_4326_raster(
            aux_src, west_4326, south_4326, east_4326, north_4326,
            bands=6, tile_size=TILE_SIZE
        )
        ndwi = read_window_from_4326_raster(
            aux_src, west_4326, south_4326, east_4326, north_4326,
            bands=7, tile_size=TILE_SIZE
        )
        ndmi = read_window_from_4326_raster(
            aux_src, west_4326, south_4326, east_4326, north_4326,
            bands=8, tile_size=TILE_SIZE
        )
        # Band 9 = ESA LULC → skipped (replaced by COSIA)
        dem = read_window_from_4326_raster(
            aux_src, west_4326, south_4326, east_4326, north_4326,
            bands=10, tile_size=TILE_SIZE
        )
    except Exception as e:
        print(f"[WARNING] aux read failed {tile_name}: {e}")
        np.save(out_path, zeros)
        saved[split] += 1
        continue

    # ── Read COSIA landcover ─────────────────────────────────────────────────
    if cosia_src is not None:
        try:
            lulc_raw = read_window_from_2154_raster(
                cosia_src,
                left_2154, bottom_2154, right_2154, top_2154,
                tile_size=TILE_SIZE,
                resampling=Resampling.nearest   # categorical — never interpolate
            )
        except Exception as e:
            print(f"[WARNING] COSIA read failed {tile_name}: {e}")
            lulc_raw = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)
    else:
        lulc_raw = np.zeros((TILE_SIZE, TILE_SIZE), dtype=np.uint8)

    # ── Normalize ────────────────────────────────────────────────────────────
    stats    = dem_stats[campaign]
    spectral = normalize_spectral(spectral)
    ndvi     = normalize_index(ndvi)
    ndwi     = normalize_index(ndwi)
    ndmi     = normalize_index(ndmi)
    dem      = normalize_dem(dem, stats["p2"], stats["p98"])

    lulc_onehot = encode_cosia_onehot(lulc_raw)   # (14, 64, 64)

    # ── Stack ────────────────────────────────────────────────────────────────
    patch = np.concatenate(
        [
            spectral,        # [0:5]    5 ch
            ndvi[None],      # [5]      1 ch
            ndwi[None],      # [6]      1 ch
            ndmi[None],      # [7]      1 ch
            lulc_onehot,     # [8:22]  14 ch
            dem[None],       # [22]     1 ch
        ],
        axis=0,
    )

    patch = np.nan_to_num(patch, nan=0.0, posinf=1.0, neginf=0.0)

    np.save(out_path, patch.astype(np.float32))
    saved[split] += 1

# ─── Cleanup ──────────────────────────────────────────────────────────────────

for src in open_aux.values():
    src.close()
for src in open_cosia.values():
    src.close()

print(f"\nDone. Saved AUX patches: {sum(saved.values())}")
print(f"  train={saved['train']}  val={saved['val']}  test={saved['test']}")
if missing["aux"]:
    print(f"  Missing aux.tif:       {sorted(missing['aux'])}")
if missing["cosia"]:
    print(f"  Missing COSIA LC:      {sorted(missing['cosia'])}")

