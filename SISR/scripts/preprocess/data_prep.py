import os
import json
import numpy as np
import rasterio
from pathlib import Path

# Paths
BASE           = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR         = BASE / "HR_downsampled"
LS_ALIGNED_DIR = BASE / "Landsat_aligned"
SPLIT_JSON     = BASE / "split_map.json"
PAIRS_JSON     = BASE / "hr_lr_pairs.json"
OUT_DIR        = BASE / "processed/patches2"

# Config 
HR_TILE       = 256
LR_TILE       = 64
STEP_TRAIN    = HR_TILE // 2   # 128px → 50% overlap
STEP_VAL_TEST = HR_TILE        # 256px → no overlap
MAX_CLOUD_FRAC   = 0.40        # skip LR tile if cloud fraction exceeds this
MAX_SHADOW_FRAC  = 0.10        # skip LR tile if shadow fraction exceeds this
MIN_VALID_FRAC   = 0.05        # skip HR tile if valid pixel fraction below this
MAX_VALID_FRAC   = 1.00        
BIT_CLOUD        = 3
BIT_CLOUD_SHADOW = 4

TEST_CAMPAIGNS = {"BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif"}

# Create output folders 
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
def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

def build_valid_mask(arr, nodata):
    valid = np.isfinite(arr)
    if nodata is None:
        return valid
    if np.isnan(nodata):
        return valid & ~np.isnan(arr)
    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return valid & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)

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

# Patching loop
all_metadata = {}
total_saved  = {"train": 0, "val": 0, "test": 0}

for hr_path in sorted(HR_DIR.glob("*.tif")):
    fname    = hr_path.name
    campaign = hr_path.stem

    if fname not in pair_lookup:
        print(f"⚠  No pair found for {fname}, skipping")
        continue

    # Verify preprocessed aligned files exist before processing
    lst_path = LS_ALIGNED_DIR / "LST_Celsius" / f"{campaign}_lst.tif"
    qa_path  = LS_ALIGNED_DIR / "QA_Pixel" / f"{campaign}_qa.tif"
    if not lst_path.exists() or not qa_path.exists():
        print(f"⚠  Preprocessed Landsat layers missing for {campaign}, skipping")
        continue

    pair_info = pair_lookup[fname]
    is_test   = fname in TEST_CAMPAIGNS

    print(f"\n{'─'*60}")
    print(f"Processing : {fname} ({'TEST' if is_test else 'TRAIN/VAL'})")

    # Define blocks
    if is_test:
        with rasterio.open(hr_path) as src:
            max_required_row = int(np.ceil(src.height / HR_TILE) * HR_TILE)
            blocks = [{"block_id": 0,
                       "row_start": 0,
                       "row_end": max_required_row,
                       "split": "test"}]
    else:
        if fname not in split_map:
            print(f"⚠  {fname} not in split_map, skipping")
            continue
        blocks = split_map[fname]["blocks"]

    # Load full arrays directly side-by-side
    with rasterio.open(hr_path) as src:
        hr_data      = src.read(1).astype(np.float32)
        hr_nodata    = src.nodata
        hr_transform = src.transform
        hr_crs       = src.crs.to_epsg()
        hr_bounds    = src.bounds

    with rasterio.open(lst_path) as src:
        lr_celsius = src.read(1).astype(np.float32)

    with rasterio.open(qa_path) as src:
        qa_data = src.read(1)

    # Mask HR nodata
    if hr_nodata is not None:
        hr_data[hr_data < -1e30] = np.nan

    # ====================================================================
    # Pad the bottom of all arrays to the newly expanded grid height
    # ====================================================================
    max_required_row = max(block["row_end"] for block in blocks)
    if hr_data.shape[0] < max_required_row:
        pad_rows = max_required_row - hr_data.shape[0]
        
        hr_data = np.pad(hr_data, ((0, pad_rows), (0, 0)), mode='constant', constant_values=np.nan)
        lr_celsius = np.pad(lr_celsius, ((0, pad_rows), (0, 0)), mode='constant', constant_values=np.nan)
        qa_data = np.pad(qa_data, ((0, pad_rows), (0, 0)), mode='constant', constant_values=0)
        
        print(f"Virtual Padding Applied: Padded bottom with {pad_rows} rows out to row {max_required_row}")

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