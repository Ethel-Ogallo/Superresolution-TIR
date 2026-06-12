import json
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from pathlib import Path

# ─── Config ──────────────────────────────────────────────────────────────────
PATCHES_BASE = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
AUX_DIR      = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/auxiliary")
COSIA_LC_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/COSIA")
STATS_PATH   = Path("/share/home/e2406751/Superresolution-TIR/data/AUX/dem_stats.json")

# Define target resolutions
CONFIGS = {
    "AUX": 64,            # LR
    "AUX_SPADE_MID": 128, # MID
    "AUX_SPADE_HR": 256   # HR
}

COSIA_CLASSES = [1, 2, 3, 4, 5, 6, 8, 9, 10, 12, 13, 14, 15, 0]
N_COSIA = len(COSIA_CLASSES)

# ─── Normalization & Encoding Helpers ────────────────────────────────────────

def get_sensor(year: int) -> str:
    return "LANDSAT" if year <= 2014 else "SENTINEL"

def normalize_spectral(x, year):
    # Fill masked NoData with 0.0
    x = np.nan_to_num(x, nan=0.0)
    
    if get_sensor(year) == 'SENTINEL':
        # Sentinel: divide by 10,000
        return np.clip(x / 10000.0, 0.0, 1.0)
    else:
        # Landsat: divide by 120.0; observed ceiling was 110
        return np.clip(x / 120.0, 0.0, 1.0)

def normalize_index(x):
    return np.clip(np.nan_to_num(x, nan=0.0), -1.0, 1.0)

def normalize_dem(x, p2, p98):
    x = np.nan_to_num(x, nan=0.0)
    return np.clip((x - p2) / (p98 - p2 + 1e-6), 0.0, 1.0)

def encode_cosia_onehot(x):
    out = np.zeros((N_COSIA, x.shape[0], x.shape[1]), dtype=np.float32)
    for i, cls in enumerate(COSIA_CLASSES):
        out[i] = (x == cls).astype(np.float32)
    return out

# ─── Main Execution ──────────────────────────────────────────────────────────

with open(PATCHES_BASE / "metadata.json") as f:
    metadata = json.load(f)
with open(STATS_PATH) as f:
    dem_stats = json.load(f)

for subfolder, size in CONFIGS.items():
    print(f"--- Processing resolution: {subfolder} ({size}x{size}) ---")
    
    for tile_name, info in metadata.items():
        campaign = info['campaign']
        year = int(campaign.split('_')[1])
        split = info['split']
        
        # Path: patches/split/resolution/tile.npy
        out_path = PATCHES_BASE / split / subfolder / f"{tile_name}.npy"
        out_path.parent.mkdir(parents=True, exist_ok=True)
        
        with rasterio.open(AUX_DIR / f"{campaign}_aux.tif") as aux, \
     rasterio.open(COSIA_LC_DIR / f"{campaign}_landcover.tif") as lc:
    
            # Define the precise HR-aligned grid using metadata bounds
            # This creates a transform for the exact patch size at the exact location
            bounds = info["bounds_2154"]
            dst_transform = rasterio.transform.from_bounds(
                bounds['left'], bounds['bottom'], bounds['right'], bounds['top'], 
                size, size
            )
            
            # Initialize destination containers
            spectral_indices = np.zeros((8, size, size), dtype=np.float32)
            dem = np.zeros((size, size), dtype=np.float32)
            lc_raw = np.zeros((size, size), dtype=np.int32)
            
            # 1. Reproject Spectral/Indices (Bands 1-8)
            reproject(
                source=rasterio.band(aux, [1, 2, 3, 4, 5, 6, 7, 8]),
                destination=spectral_indices,
                src_transform=aux.transform, src_crs=aux.crs,
                dst_transform=dst_transform, dst_crs="EPSG:2154",
                resampling=Resampling.bilinear
            )
            
            # 2. Reproject DEM (Band 10)
            reproject(
                source=rasterio.band(aux, 10),
                destination=dem,
                src_transform=aux.transform, src_crs=aux.crs,
                dst_transform=dst_transform, dst_crs="EPSG:2154",
                resampling=Resampling.bilinear
            )
            
            # 3. Reproject Landcover (Nearest neighbor is crucial for categorical data)
            reproject(
                source=rasterio.band(lc, 1),
                destination=lc_raw,
                src_transform=lc.transform, src_crs=lc.crs,
                dst_transform=dst_transform, dst_crs="EPSG:2154",
                resampling=Resampling.nearest
            )
                        
            # 3. Apply Normalization
            spectral_indices[:5] = normalize_spectral(spectral_indices[:5], year)
            spectral_indices[5:] = normalize_index(spectral_indices[5:])
            dem = normalize_dem(dem, dem_stats[campaign]["p2"], dem_stats[campaign]["p98"])
            
            # 4. Stack and Save
            final = np.concatenate([spectral_indices, encode_cosia_onehot(lc_raw), dem[None]], axis=0)
            np.save(out_path, final.astype(np.float32))

print("Auxiliary patching complete for all resolutions.")

# Channel mapping: 0-4: Spectral, 5: NDVI, 6: NDWI, 7: NDMI, 8-21: LC, 22: DEM