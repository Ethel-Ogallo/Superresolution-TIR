import os
import json
import tarfile
import tempfile
import shutil
import re
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path

# Paths
BASE       = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR     = BASE / "HR_downsampled"
LS_DIR     = BASE / "Landsat"
SPLIT_JSON = BASE / "split_map.json"
PAIRS_JSON = BASE / "hr_lr_pairs.json"
OUT_DIR    = BASE / "processed/patches"

# Config 
HR_TILE       = 256
LR_TILE       = 64
STEP_TRAIN    = HR_TILE // 2   # 128px → 50% overlap
STEP_VAL_TEST = HR_TILE        # 256px → no overlap
CLIP_BUFFER_M = 300            # buffer when clipping Landsat to HR footprint
MAX_CLOUD_FRAC   = 0.40        # skip LR tile if cloud fraction exceeds this
MAX_SHADOW_FRAC  = 0.10        # skip LR tile if shadow fraction exceeds this
MIN_VALID_FRAC   = 0.10        # skip HR tile if valid pixel fraction below this
MAX_VALID_FRAC   = 1.00        
BIT_CLOUD        = 3
BIT_CLOUD_SHADOW = 4

TEST_CAMPAIGNS = {"BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif"}

#  Create output folders 
for split in ["train", "val", "test"]:
    for res in ["HR", "LR"]:
        (OUT_DIR / split / res).mkdir(parents=True, exist_ok=True)

# Load JSONs
with open(SPLIT_JSON) as f:
    split_map = json.load(f)

with open(PAIRS_JSON) as f:
    pairs_data = json.load(f)

# Build lookup: hr_name → selected landsat info
pair_lookup = {}
for p in pairs_data["pairs"]:
    selected = next((c for c in p["candidates"] if c["selected"]), None)
    if selected:
        pair_lookup[p["hr_name"]] = {
            "ls_name":   selected["ls_name"],
            "hr_date":   p["hr_date"],
            "ls_date":   selected["ls_date"],
            "satellite": selected["satellite"],
            "date_gap":  selected["date_gap_days"]
        }

# -------------- Helpers --------------
# parse MTL numeric fields 
def parse_mtl_numeric(path):
    """Read scale/offset and other numeric fields from MTL text file."""
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"')
                try:
                    data[k] = float(v)
                except ValueError:
                    continue
    return data

# extract bit from QA array
def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

# build valid mask 
def build_valid_mask(arr, nodata):
    valid = np.isfinite(arr)
    if nodata is None:
        return valid
    if np.isnan(nodata):
        return valid & ~np.isnan(arr)
    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return valid & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)

# extract Band10 + QA + MTL from tar 
def extract_scene_files(tar_path, tmp_dir):
    """Extract ST_B10, QA_PIXEL and MTL from tar. Returns (b10, qa, mtl) paths."""
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)

    files = list(tmp_dir.iterdir())
    b10 = next((f for f in files if "ST_B10" in f.name and f.suffix == ".TIF"), None)
    qa  = next((f for f in files if "QA_PIXEL" in f.name and f.suffix == ".TIF"), None)
    mtl = next((f for f in files if "_MTL.txt" in f.name), None)

    if not b10 or not qa or not mtl:
        raise FileNotFoundError(
            f"Missing files in {tar_path.name}. "
            f"Found: {[f.name for f in files]}"
        )
    return b10, qa, mtl

# clip raster to HR footprint 
def clip_to_hr(input_path, hr_path, output_path, resampling=Resampling.bilinear):
    """Clip and reproject input raster to HR footprint + buffer."""
    with rasterio.open(hr_path) as ref:
        hr_crs    = ref.crs
        hr_bounds = ref.bounds
        hr_transform = ref.transform
        hr_width  = ref.width
        hr_height = ref.height

    with rasterio.open(input_path) as src:
        # Buffer HR bbox and clip
        bbox = box(*hr_bounds)
        gdf  = gpd.GeoDataFrame(geometry=[bbox], crs=hr_crs)
        geom = [gdf.to_crs(src.crs).buffer(CLIP_BUFFER_M).geometry.values[0].__geo_interface__]
        clipped, clip_transform = mask(src, geom, crop=True)
        clip_meta = src.meta.copy()
        clip_meta.update({
            "height": clipped.shape[1],
            "width":  clipped.shape[2],
            "transform": clip_transform,
            "dtype": "float32"
        })

        # Write clipped
        tmp_clip = output_path.parent / f"_tmp_clip_{output_path.name}"
        with rasterio.open(tmp_clip, "w", **clip_meta) as tmp:
            tmp.write(clipped.astype(np.float32))

    # Reproject clipped to HR grid
    with rasterio.open(tmp_clip) as src:
        dst_data = np.zeros((hr_height, hr_width), dtype=np.float32)
        reproject(
            source=rasterio.band(src, 1),
            destination=dst_data,
            src_transform=src.transform,
            src_crs=src.crs,
            dst_transform=hr_transform,
            dst_crs=hr_crs,
            resampling=resampling
        )

    out_meta = {
        "driver": "GTiff", "height": hr_height, "width": hr_width,
        "count": 1, "dtype": "float32",
        "crs": hr_crs, "transform": hr_transform
    }
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(dst_data, 1)

    tmp_clip.unlink()

# extract LR tile at 64x64 
def extract_lr_tile(lr_data, row, col):
    """Block-average 256x256 region of LR (at HR grid) down to 64x64."""
    region = lr_data[row:row+HR_TILE, col:col+HR_TILE]
    h, w   = region.shape
    h4, w4 = (h // 4) * 4, (w // 4) * 4
    region = region[:h4, :w4]
    return region.reshape(h4//4, 4, w4//4, 4).mean(axis=(1, 3)).astype(np.float32)

def extract_qa_tile(qa_data, row, col):
    """Extract QA tile at LR resolution (direct pixel lookup, no averaging)."""
    return qa_data[row:row+HR_TILE:4, col:col+HR_TILE:4]

# -------------- Patch logic --------------
# DEBUG — remove after testing
# DEBUG_SINGLE = "BAS_2025.tif"  # train/val campaign
# DEBUG_SINGLE = "BRC_2022.tif"  # test campaign


all_metadata = {}
total_saved  = {"train": 0, "val": 0, "test": 0}

for hr_path in sorted(HR_DIR.glob("*.tif")):
    fname    = hr_path.name
    campaign = hr_path.stem
    # if fname != DEBUG_SINGLE:  # DEBUG — remove after testing
    #     continue

    if fname not in pair_lookup:
        print(f"⚠  No pair found for {fname}, skipping")
        continue
   

    pair_info = pair_lookup[fname]
    ls_tar    = LS_DIR / pair_info["ls_name"]
    is_test   = fname in TEST_CAMPAIGNS

    print(f"\n{'─'*60}")
    print(f"Processing : {fname} ({'TEST' if is_test else 'TRAIN/VAL'})")
    print(f"Paired with: {pair_info['ls_name']}")

    # Define blocks
    if is_test:
        with rasterio.open(hr_path) as src:
            blocks = [{"block_id": 0,
                       "row_start": 0,
                       "row_end": src.height,
                       "split": "test"}]
    else:
        if fname not in split_map:
            print(f"⚠  {fname} not in split_map, skipping")
            continue
        blocks = split_map[fname]["blocks"]

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        print(f"  Extracting tar...")
        b10_path, qa_path, mtl_path = extract_scene_files(ls_tar, tmp_dir)

        # Read scale/offset from MTL
        mtl_data = parse_mtl_numeric(mtl_path)
        scale    = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"]
        offset   = mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]
        print(f"  MTL scale={scale}, offset={offset}")

        # Clip + reproject B10 and QA to HR grid
        print(f"  Aligning Band10 to HR grid...")
        b10_aligned = tmp_dir / "b10_aligned.tif"
        qa_aligned  = tmp_dir / "qa_aligned.tif"
        clip_to_hr(b10_path, hr_path, b10_aligned, Resampling.bilinear)
        clip_to_hr(qa_path,  hr_path, qa_aligned,  Resampling.nearest)

        # Load full arrays
        with rasterio.open(hr_path) as src:
            hr_data      = src.read(1).astype(np.float32)
            hr_nodata    = src.nodata
            hr_transform = src.transform
            hr_crs       = src.crs.to_epsg()
            hr_bounds    = src.bounds

        with rasterio.open(b10_aligned) as src:
            b10_data = src.read(1).astype(np.float32)

        with rasterio.open(qa_aligned) as src:
            qa_data = src.read(1)

        # Mask HR nodata
        if hr_nodata is not None:
            hr_data[hr_data < -1e30] = np.nan

        # Convert B10 DN → Celsius using MTL values
        b10_valid = (b10_data > 0)
        lr_celsius = np.where(b10_valid,
                              b10_data * scale + offset - 273.15,
                              np.nan).astype(np.float32)

        # ── Tile block by block ──
        for block in blocks:
            split     = block["split"]
            block_id  = block["block_id"]
            row_start = block["row_start"]
            row_end   = block["row_end"]
            step      = STEP_TRAIN if split == "train" else STEP_VAL_TEST

            n_saved = 0
            n_skip_nodata = 0
            n_skip_cloud  = 0

            for r in range(row_start, row_end, step):
                if r + HR_TILE > row_end:
                    continue  # no tile bleeds across block boundary
                for c in range(0, hr_data.shape[1], step):
                    if c + HR_TILE > hr_data.shape[1]:
                        continue  # no tile bleeds beyond raster width

                    hr_tile = hr_data[r:r+HR_TILE, c:c+HR_TILE]
                    lr_tile = extract_lr_tile(lr_celsius, r, c)
                    qa_tile = extract_qa_tile(qa_data, r, c)

                    # ── Valid pixel check on HR tile ──
                    hr_valid_mask = build_valid_mask(hr_tile, hr_nodata)
                    hr_valid_frac = hr_valid_mask.mean()
                    if not (MIN_VALID_FRAC <= hr_valid_frac <= MAX_VALID_FRAC):
                        n_skip_nodata += 1
                        continue

                    # ── Cloud/shadow check on LR tile ──
                    cloud_frac  = extract_bit(qa_tile, BIT_CLOUD).mean()
                    shadow_frac = extract_bit(qa_tile, BIT_CLOUD_SHADOW).mean()
                    if cloud_frac > MAX_CLOUD_FRAC or shadow_frac > MAX_SHADOW_FRAC:
                        n_skip_cloud += 1
                        continue

                    # ── Valid pixel check on LR tile ──
                    lr_valid_frac = np.isfinite(lr_tile).mean()
                    if lr_valid_frac < MIN_VALID_FRAC:
                        n_skip_nodata += 1
                        continue

                    tile_name = f"{campaign}_b{block_id}_r{r}_c{c}"

                    np.save(OUT_DIR / split / "HR" / f"{tile_name}.npy", hr_tile)
                    np.save(OUT_DIR / split / "LR" / f"{tile_name}.npy", lr_tile)

                    # Tile bounds in EPSG:2154
                    left   = hr_bounds.left + c * 7.5
                    top    = hr_bounds.top  - r * 7.5
                    right  = left + HR_TILE * 7.5
                    bottom = top  - HR_TILE * 7.5

                    all_metadata[tile_name] = {
                        "campaign":       campaign,
                        "split":          split,
                        "block_id":       block_id,
                        "row_origin":     r,
                        "col_origin":     c,
                        "hr_source":      fname,
                        "lr_source":      pair_info["ls_name"],
                        "hr_date":        pair_info["hr_date"],
                        "lr_date":        pair_info["ls_date"],
                        "satellite":      pair_info["satellite"],
                        "date_gap_days":  pair_info["date_gap"],
                        "crs":            f"EPSG:{hr_crs}",
                        "step_used":      step,
                        "hr_valid_frac":  round(float(hr_valid_frac), 4),
                        "lr_valid_frac":  round(float(lr_valid_frac), 4),
                        "cloud_frac":     round(float(cloud_frac), 4),
                        "shadow_frac":    round(float(shadow_frac), 4),
                        "bounds_2154": {
                            "left":   left,
                            "right":  right,
                            "top":    top,
                            "bottom": bottom
                        }
                    }
                    n_saved += 1

            total_saved[split] += n_saved
            print(f"  Block {block_id} ({split}, step={step}): "
                  f"{n_saved} saved | "
                  f"{n_skip_nodata} skipped nodata | "
                  f"{n_skip_cloud} skipped cloud")

    finally:
        shutil.rmtree(tmp_dir)
        print(f"  Temp files cleaned up")

# Save metadata
meta_path = OUT_DIR / "metadata.json"
with open(meta_path, "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"\n{'═'*60}")
print(f"Done.")
print(f"  Train tiles : {total_saved['train']}")
print(f"  Val tiles   : {total_saved['val']}")
print(f"  Test tiles  : {total_saved['test']}")
print(f"  Metadata    : {meta_path}")