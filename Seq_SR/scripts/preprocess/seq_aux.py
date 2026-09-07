import json
import numpy as np
import rasterio
import h5py
from rasterio.warp import reproject, Resampling
from pathlib import Path


# paths
BASE = Path("/share/home/e2406751/Superresolution-TIR/data")
SEQ_DIR = BASE / "processed/seq_patches2"
AUX_DIR = BASE / "AUX/auxiliary"
COSIA_DIR = BASE / "AUX/COSIA"
STATS_PATH = BASE / "AUX/dem_stats.json"

# config
LR_TILE = 64
COSIA_CLASSES = [1,2,3,4,5,6,8,9,10,12,13,14,15,0]
N_COSIA = len(COSIA_CLASSES)
N_AUX_CH = 8 + N_COSIA + 1  # spectral+indices (8) + landcover onehot + dem = 23

# jsons
with open(SEQ_DIR / "frame_metadata.json") as f:
    frame_metadata = json.load(f)
with open(STATS_PATH) as f:
    dem_stats = json.load(f)

# helper functions
def clean_and_clip_spectral(x): 
    return np.clip(np.nan_to_num(x, nan=0.0), 0.0, 1.0)

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

# ----------------- Preprocess Auxiliary Data -----------------
# n_frames = len(frame_metadata)
# all_aux = np.zeros((n_frames, N_AUX_CH, LR_TILE, LR_TILE), dtype=np.float32)

# for fid, meta in frame_metadata.items():
#     frame_idx = int(fid.replace("f", ""))
#     campaign = meta["hr_source"].replace(".tif", "")
#     b = meta["bounds_2154"]

#     with rasterio.open(AUX_DIR / f"{campaign}_aux.tif") as aux, \
#          rasterio.open(COSIA_DIR / f"{campaign}_landcover.tif") as lc:

#         dst_transform = rasterio.transform.from_bounds(
#             b["left"], b["bottom"], b["right"], b["top"], LR_TILE, LR_TILE
#         )

#         spectral_indices = np.zeros((8, LR_TILE, LR_TILE), dtype=np.float32)
#         dem = np.zeros((LR_TILE, LR_TILE), dtype=np.float32)
#         lc_raw = np.zeros((LR_TILE, LR_TILE), dtype=np.int32)

#         reproject(source=rasterio.band(aux, [1,2,3,4,5,6,7,8]), destination=spectral_indices,
#                   src_transform=aux.transform, src_crs=aux.crs,
#                   dst_transform=dst_transform, dst_crs="EPSG:2154", resampling=Resampling.bilinear)
#         reproject(source=rasterio.band(aux, 9), destination=dem,
#                   src_transform=aux.transform, src_crs=aux.crs,
#                   dst_transform=dst_transform, dst_crs="EPSG:2154", resampling=Resampling.bilinear)
#         reproject(source=rasterio.band(lc, 1), destination=lc_raw,
#                   src_transform=lc.transform, src_crs=lc.crs,
#                   dst_transform=dst_transform, dst_crs="EPSG:2154", resampling=Resampling.nearest)

#         spectral_indices[:5] = clean_and_clip_spectral(spectral_indices[:5])
#         spectral_indices[5:] = normalize_index(spectral_indices[5:])
#         dem = normalize_dem(dem, dem_stats[campaign]["p2"], dem_stats[campaign]["p98"])

#         final = np.concatenate([spectral_indices, encode_cosia_onehot(lc_raw), dem[None]], axis=0)
#         all_aux[frame_idx] = final

# with h5py.File(SEQ_DIR / "aux_frames.h5", "w") as hf:
#     hf.create_dataset("frames", data=all_aux, compression="gzip", compression_opts=4)

# print(f"Saved aux_frames.h5 with shape {all_aux.shape}")



#### SPADE landcover patches
HR_TILE  = 256
MID_TILE = HR_TILE // 2   # 128 -> matches upconv1 output resolution

def reproject_cosia_onehot(lc_src, bounds, tile_size):
    dst_transform = rasterio.transform.from_bounds(
        bounds["left"], bounds["bottom"], bounds["right"], bounds["top"],
        tile_size, tile_size,
    )
    lc_raw = np.zeros((tile_size, tile_size), dtype=np.int32)
    reproject(
        source=rasterio.band(lc_src, 1), destination=lc_raw,
        src_transform=lc_src.transform, src_crs=lc_src.crs,
        dst_transform=dst_transform, dst_crs="EPSG:2154",
        resampling=Resampling.nearest,
    )
    return encode_cosia_onehot(lc_raw)


# ----------------- Build aux_mid / aux_hr -----------------
n_frames = len(frame_metadata)
all_mid = np.zeros((n_frames, N_COSIA, MID_TILE, MID_TILE), dtype=np.float32)
all_hr  = np.zeros((n_frames, N_COSIA, HR_TILE,  HR_TILE),  dtype=np.float32)

for fid, meta in frame_metadata.items():
    frame_idx = int(fid.replace("f", ""))
    campaign = meta["hr_source"].replace(".tif", "")
    b = meta["bounds_2154"]  # same HR-tile bounds used to build the HR TIR frames

    with rasterio.open(COSIA_DIR / f"{campaign}_landcover.tif") as lc:
        all_hr[frame_idx]  = reproject_cosia_onehot(lc, b, HR_TILE)
        all_mid[frame_idx] = reproject_cosia_onehot(lc, b, MID_TILE)

with h5py.File(SEQ_DIR / "aux_mid_frames.h5", "w") as hf:
    hf.create_dataset("frames", data=all_mid, compression="gzip", compression_opts=4)
with h5py.File(SEQ_DIR / "aux_hr_frames.h5", "w") as hf:
    hf.create_dataset("frames", data=all_hr, compression="gzip", compression_opts=4)

print(f"Saved aux_mid_frames.h5 with shape {all_mid.shape}")
print(f"Saved aux_hr_frames.h5 with shape {all_hr.shape}")