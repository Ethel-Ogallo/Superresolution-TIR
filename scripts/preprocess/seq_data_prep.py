"""
build_seq_patches.py
"""

import json
import tarfile
import tempfile
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path
from scipy.ndimage import label

# ============================================================
# Paths
# ============================================================
BASE       = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR     = BASE / "HR_downsampled"
LS_DIR     = BASE / "Landsat"
SPLIT_JSON = BASE / "seq_split_map.json"
PAIRS_JSON = BASE / "hr_lr_pairs.json"
OUT_DIR    = BASE / "processed/seq_patches"

# ============================================================
# Config
# ============================================================
HR_TILE          = 256
LR_TILE          = 64
SEQ_LEN          = 5            # Frames per VSR sequence
CLIP_BUFFER_M    = 300
MAX_CLOUD_FRAC   = 0.40
MAX_SHADOW_FRAC  = 0.10
MIN_VALID_FRAC   = 0.10
MAX_VALID_FRAC   = 1.00
BIT_CLOUD        = 3
BIT_CLOUD_SHADOW = 4

# Spatial sampling strategy
# Training uses overlap to increase sample density.
# Validation/Test use non-overlapping patches.
TRAIN_PATCH_STEP = HR_TILE // 2   # 128 pixels = 50% overlap
EVAL_PATCH_STEP  = HR_TILE        # 256 pixels = no overlap

PATCH_VALID_THRESHOLD = MIN_VALID_FRAC

TEST_CAMPAIGNS = {"BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif"}

# Create structural output directories
for split in ["train", "val", "test"]:
    (OUT_DIR / split).mkdir(parents=True, exist_ok=True)

# Load configuration databases
with open(SPLIT_JSON) as f:
    split_map = json.load(f)

with open(PAIRS_JSON) as f:
    pairs_data = json.load(f)

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

# ============================================================
# Core helper utilities
# ============================================================
def parse_mtl_numeric(path):
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=", 1)
                k, v = k.strip(), v.strip().strip('"')
                try: data[k] = float(v)
                except ValueError: continue
    return data

def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

def build_valid_mask(arr, nodata):
    valid = np.isfinite(arr)
    if nodata is None: return valid
    if np.isnan(nodata): return valid & ~np.isnan(arr)
    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return valid & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)

def extract_scene_files(tar_path, tmp_dir):
    with tarfile.open(tar_path) as tar:
        tar.extractall(tmp_dir)
    files = list(tmp_dir.iterdir())
    b10 = next((f for f in files if "ST_B10" in f.name and f.suffix == ".TIF"), None)
    qa  = next((f for f in files if "QA_PIXEL" in f.name and f.suffix == ".TIF"), None)
    mtl = next((f for f in files if "_MTL.txt" in f.name), None)
    if not b10 or not qa or not mtl:
        raise FileNotFoundError(f"Missing core bands in {tar_path.name}")
    return b10, qa, mtl

def clip_to_hr(input_path, hr_path, output_path, resampling=Resampling.bilinear):
    with rasterio.open(hr_path) as ref:
        hr_crs, hr_bounds, hr_transform, hr_width, hr_height = ref.crs, ref.bounds, ref.transform, ref.width, ref.height

    with rasterio.open(input_path) as src:
        bbox = box(*hr_bounds)
        gdf  = gpd.GeoDataFrame(geometry=[bbox], crs=hr_crs)
        geom = [gdf.to_crs(src.crs).buffer(CLIP_BUFFER_M).geometry.values[0].__geo_interface__]
        clipped, clip_transform = mask(src, geom, crop=True)
        clip_meta = src.meta.copy()
        clip_meta.update({"height": clipped.shape[1],
                          "width": clipped.shape[2],
                          "transform": clip_transform,
                          "dtype": "float32"})

        tmp_clip = output_path.parent / f"_tmp_clip_{output_path.name}"
        with rasterio.open(tmp_clip, "w", **clip_meta) as tmp:
            tmp.write(clipped.astype(np.float32))

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

    out_meta = {"driver": "GTiff",
                "height": hr_height,
                "width": hr_width,
                "count": 1,
                "dtype": "float32",
                "crs": hr_crs,
                "transform": hr_transform}

    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(dst_data, 1)

    tmp_clip.unlink()

def extract_lr_tile(lr_data, row, col):
    region = lr_data[row:row+HR_TILE, col:col+HR_TILE]
    h, w   = region.shape
    h4, w4 = (h // 4) * 4, (w // 4) * 4
    region = region[:h4, :w4]
    return region.reshape(h4//4, 4, w4//4, 4).mean(axis=(1, 3)).astype(np.float32)

def extract_qa_tile(qa_data, row, col):
    return qa_data[row:row+HR_TILE:4, col:col+HR_TILE:4]

def make_tile_or_none(r, c, hr_data, lr_celsius, qa_data, hr_nodata):

    hr_tile = hr_data[r:r+HR_TILE, c:c+HR_TILE]
    lr_tile = extract_lr_tile(lr_celsius, r, c)
    qa_tile = extract_qa_tile(qa_data, r, c)
   
    hr_valid_mask = build_valid_mask(hr_tile, hr_nodata)

    if not (
        MIN_VALID_FRAC <= hr_valid_mask.mean() <= MAX_VALID_FRAC
    ) or (
        np.isfinite(lr_tile).mean() < MIN_VALID_FRAC
    ):
        return None, "nodata"


    if (
        extract_bit(qa_tile, BIT_CLOUD).mean() > MAX_CLOUD_FRAC
        or
        extract_bit(qa_tile, BIT_CLOUD_SHADOW).mean() > MAX_SHADOW_FRAC
    ):
        return None, "cloud"


    return {
        "hr": hr_tile,
        "lr": lr_tile
    }, None

def assign_split(row, blocks):
    for block in blocks:
        if block["row_start"] <= row < block["row_end"]:
            return block["split"], block["block_id"]

    return None, None

# ============================================================
# Spatial patch ordering helpers
# ============================================================

def generate_candidate_patches(valid_mask, row_start, row_end, patch_step):
    """
    Generate HR patch locations over valid thermal footprint.

    Training:
        Uses overlapping patches.

    Validation/Test:
        Uses non-overlapping patches to reduce dependency between
        evaluation samples.

    All valid pixels are retained; no river-only masking is applied.
    """
    patches = []
    height, width = valid_mask.shape

    for r in range(
        row_start,
        row_end - HR_TILE + 1,
        patch_step
    ):

        for c in range(
            0,
            width - HR_TILE + 1,
            patch_step
        ):

            footprint = valid_mask[
                r:r+HR_TILE,
                c:c+HR_TILE
            ]

            if footprint.mean() >= PATCH_VALID_THRESHOLD:
                patches.append(
                    {
                        "row_origin": r,
                        "col_origin": c
                    }
                )

    return patches


def order_patches_north_south(patches):
    """
    Orders spatial patches into a physically coherent downstream sequence.

    Assumption:
        Raster north-south direction approximates river upstream-downstream
        direction.

    Behaviour:
        - Starts from northernmost patch.
        - Moves progressively south.
        - Allows east/west movement to follow bends.
        - Prevents jumps across disconnected areas.
    """

    if len(patches) == 0:
        return []

    remaining = patches.copy()

    # start upstream / northernmost
    current = min(
        remaining,
        key=lambda p: (
            p["row_origin"],
            p["col_origin"]
        )
    )

    remaining.remove(current)

    ordered = [current]

    MAX_SPATIAL_JUMP = HR_TILE * 2.5   # ~640 pixels maximum movement

    while remaining:
        candidates = []

        for idx, p in enumerate(remaining):

            dr = p["row_origin"] - current["row_origin"]
            dc = p["col_origin"] - current["col_origin"]

            distance = np.sqrt(dr**2 + dc**2)

            # reject very large jumps
            if distance > MAX_SPATIAL_JUMP:
                continue

            # Prefer downstream movement
            # Positive dr = moving south in raster coordinates
            if dr >= 0:
                direction_penalty = 0
            else:
                direction_penalty = HR_TILE * 3

            # Total score:
            # close patches preferred
            # southward preferred
            score = distance + direction_penalty

            candidates.append((score,idx))

        # If we are at a bend/loop and no southward patch exists,
        # take the closest available patch rather than breaking.
        if len(candidates) == 0:

            distances = []

            for idx, p in enumerate(remaining):

                dr = p["row_origin"] - current["row_origin"]
                dc = p["col_origin"] - current["col_origin"]

                distances.append((np.sqrt(dr**2 + dc**2),idx))

            _, idx = min(distances)

        else:
            _, idx = min(candidates)


        current = remaining.pop(idx)
        ordered.append(current)

    return ordered


# ============================================================
# SR sequence assembly
# ============================================================
def build_sequences_from_track(track, campaign, block_id, pair_info, hr_bounds, hr_crs, fname):
    """Slices a continuous track of patches into SEQ_LEN sequences based on split rules."""
    global total_saved
    i = 0
    split = track[0]['split']
    # seq_step = 1 if split == "train" else SEQ_LEN
    if split == "train":
        seq_step = 1

    elif split == "val":
        seq_step = 1

    elif split == "test":
        seq_step = SEQ_LEN
        
    n_saved = 0

    while i <= len(track) - SEQ_LEN:
        sub_seq = track[i: i + SEQ_LEN]

        splits_in_seq = [p['split'] for p in sub_seq]
        if len(set(splits_in_seq)) != 1:
            i += 1
            continue

        start_r = sub_seq[0]['row_origin']
        start_c = sub_seq[0]['col_origin']
        seq_name = f"{campaign}_b{block_id}_r{start_r}_c{start_c}"

        hr_seq_dir = OUT_DIR / split / "HR" / seq_name
        lr_seq_dir = OUT_DIR / split / "LR" / seq_name

        hr_seq_dir.mkdir(parents=True, exist_ok=True)
        lr_seq_dir.mkdir(parents=True, exist_ok=True)

        for frame_idx, frame_data in enumerate(sub_seq):
            np.save(hr_seq_dir / f"{frame_idx:04d}.npy", frame_data['hr'])
            np.save(lr_seq_dir / f"{frame_idx:04d}.npy", frame_data['lr'])

            left   = hr_bounds.left + frame_data['col_origin'] * 7.5
            top    = hr_bounds.top  - frame_data['row_origin'] * 7.5
            right  = left + HR_TILE * 7.5
            bottom = top  - HR_TILE * 7.5

            all_metadata[f"{seq_name}_f{frame_idx:04d}"] = {
                "sequence_group":  seq_name,
                "frame_index":     frame_idx,
                "campaign":        campaign,
                "split":           split,
                "block_id":        block_id,
                "row_origin":      frame_data['row_origin'],
                "col_origin":      frame_data['col_origin'],
                "hr_source":       fname,
                "lr_source":       pair_info["ls_name"],
                "hr_date":         pair_info["hr_date"],
                "lr_date":         pair_info["ls_date"],
                "satellite":       pair_info["satellite"],
                "date_gap_days":   pair_info["date_gap"],
                "bounds_2154":     {"left": left, "right": right, "top": top, "bottom": bottom}
            }

        n_saved += 1
        total_saved[split] += 1
        i += seq_step
    return n_saved


# ============================================================
# Main processing pipeline
# ============================================================
all_metadata = {}
total_saved  = {"train": 0, "val": 0, "test": 0}

for hr_path in sorted(HR_DIR.glob("*.tif")):
    fname, campaign = hr_path.name, hr_path.stem

    if fname not in pair_lookup:
        print(f"No pair found for {fname}, skipping")
        continue

    pair_info = pair_lookup[fname]
    ls_tar    = LS_DIR / pair_info["ls_name"]
    is_test   = fname in TEST_CAMPAIGNS

    print(f"\n{'='*60}\nProcessing Stream: {fname} ({'TEST' if is_test else 'TRAIN/VAL'})")

    tmp_dir = Path(tempfile.mkdtemp())
    try:
        b10_path, qa_path, mtl_path = extract_scene_files(ls_tar, tmp_dir)
        mtl_data = parse_mtl_numeric(mtl_path)
        scale, offset = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"], mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]

        b10_aligned, qa_aligned = tmp_dir / "b10_aligned.tif", tmp_dir / "qa_aligned.tif"
        clip_to_hr(b10_path, hr_path, b10_aligned, Resampling.bilinear)
        clip_to_hr(qa_path,  hr_path, qa_aligned,  Resampling.nearest)

        with rasterio.open(hr_path) as src:
            hr_data, hr_nodata, hr_crs, hr_bounds = src.read(1).astype(np.float32), src.nodata, src.crs.to_epsg(), src.bounds
        with rasterio.open(b10_aligned) as src: b10_data = src.read(1).astype(np.float32)
        with rasterio.open(qa_aligned) as src:  qa_data  = src.read(1)

        if hr_nodata is not None: hr_data[hr_data < -1e30] = np.nan
        b10_valid = (b10_data > 0)
        lr_celsius = np.where(b10_valid, b10_data * scale + offset - 273.15, np.nan).astype(np.float32)

        # Precomputed once per campaign -- this IS the flight/river corridor
        # footprint. The centerline tracker below follows this, not a fixed grid.
        hr_valid_full = build_valid_mask(hr_data, hr_nodata)

        blocks = [{"block_id": 0, "row_start": 0, "row_end": hr_data.shape[0], "split": "test"}] if is_test else split_map[fname]["blocks"]

        for block in blocks:
            split, block_id, row_start, row_end = block["split"], block["block_id"], block["row_start"], block["row_end"]

            n_block_sequences = 0
            n_skip_nodata = 0
            n_skip_cloud  = 0
            
            # ------------------------------------------------------------
            # Generate spatially ordered patch track
            # ------------------------------------------------------------

            if split == "train":
                patch_step = TRAIN_PATCH_STEP
            else:
                patch_step = EVAL_PATCH_STEP


            patch_locations = generate_candidate_patches(
                hr_valid_full,
                row_start,
                row_end,
                patch_step
            )

            patch_locations = order_patches_north_south(
                patch_locations
            )

            track = []

            for loc in patch_locations:
                r = loc["row_origin"]
                c = loc["col_origin"]

                tile, skip_reason = make_tile_or_none(
                    r,
                    c,
                    hr_data,
                    lr_celsius,
                    qa_data,
                    hr_nodata
                )

                if tile is None:
                    if skip_reason == "cloud":
                        n_skip_cloud += 1
                    else:
                        n_skip_nodata += 1

                    continue

                track.append(
                    {
                        "row_origin": r,
                        "col_origin": c,
                        "split": split,
                        **tile
                    }
                )

            if len(track) >= SEQ_LEN:
                n_block_sequences += build_sequences_from_track(
                    track,
                    campaign,
                    block_id,
                    pair_info,
                    hr_bounds,
                    hr_crs,
                    fname
                )

            print(f"  Block {block_id} ({split}): {n_block_sequences} SR sequences built | Skipped: {n_skip_nodata} nodata, {n_skip_cloud} clouds")

    finally:
        shutil.rmtree(tmp_dir)

# Export consolidated metadata log
with open(OUT_DIR / "metadata.json", "w") as f:
    json.dump(all_metadata, f, indent=2)

print(f"\nPatching complete;\nTotal sequences saved: {sum(total_saved.values())}")
print(f"Train Sequences : {total_saved['train']}")
print(f"Val Sequences   : {total_saved['val']}")
print(f"Test Sequences  : {total_saved['test']}")