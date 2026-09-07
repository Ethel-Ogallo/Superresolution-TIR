"""
count_seq_yield.py

Answers one question before you commit to a patch_step / seq_step config:
"how many VSR sequences does each split/block/campaign actually produce?"

Reuses the exact same raster loading, valid-mask, cloud-filtering, and
patch-ordering logic as build_seq_patches.py (copied verbatim, not
reimplemented) so the counts you get here match what a real run would
produce -- this does NOT save any .npy files or touch your existing
processed/seq_patches directory. It's read-only diagnostics.

Sweeps:
  - TRAIN candidate: patch_step in TRAIN_STEP_CANDIDATES
  - EVAL  candidate: patch_step in EVAL_STEP_CANDIDATES  (applies to val/test)
  - seq_step for val is fixed to SEQ_LEN (the bugfix), test always SEQ_LEN,
    train always 1 -- these aren't swept since we already settled them.

Output: a table per campaign/block/split showing raw track length (number
of valid ordered patches) and resulting sequence count for each patch_step
candidate, plus split-level totals so you can see whether val/test will
have enough sequences to trust water_mae as a stable metric.
"""

import json
import tempfile
import shutil
import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd
from pathlib import Path

# ============================================================
# Paths -- must match build_seq_patches.py
# ============================================================
BASE       = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR     = BASE / "HR_downsampled"
LS_DIR     = BASE / "Landsat"
SPLIT_JSON = BASE / "seq_split_map.json"
PAIRS_JSON = BASE / "hr_lr_pairs.json"

HR_TILE          = 256
SEQ_LEN          = 5
CLIP_BUFFER_M    = 300
MAX_CLOUD_FRAC   = 0.40
MAX_SHADOW_FRAC  = 0.10
MIN_VALID_FRAC   = 0.10
MAX_VALID_FRAC   = 1.00
BIT_CLOUD        = 3
BIT_CLOUD_SHADOW = 4
PATCH_VALID_THRESHOLD = MIN_VALID_FRAC
TEST_CAMPAIGNS = {"BRC_2022.tif", "BRC_2023.tif", "HAUT_2025.tif"}

# What to sweep -- edit these if you want other candidates
TRAIN_STEP_CANDIDATES = [HR_TILE // 2, HR_TILE]        # 128, 256
EVAL_STEP_CANDIDATES  = [HR_TILE // 2, HR_TILE]         # 128, 256

with open(SPLIT_JSON) as f:
    split_map = json.load(f)
with open(PAIRS_JSON) as f:
    pairs_data = json.load(f)
pair_lookup = {}
for p in pairs_data["pairs"]:
    selected = next((c for c in p["candidates"] if c["selected"]), None)
    if selected:
        pair_lookup[p["hr_name"]] = {
            "ls_name": selected["ls_name"],
            "date_gap": selected["date_gap_days"],
        }


# ============================================================
# Copied verbatim from build_seq_patches.py (logic must match exactly)
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


def build_valid_mask(arr, nodata):
    valid = np.isfinite(arr)
    if nodata is None: return valid
    if np.isnan(nodata): return valid & ~np.isnan(arr)
    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return valid & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)


def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1


def extract_scene_files(tar_path, tmp_dir):
    import tarfile
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
        hr_crs, hr_bounds, hr_transform = ref.crs, ref.bounds, ref.transform
        hr_width, hr_height = ref.width, ref.height
    with rasterio.open(input_path) as src:
        bbox = box(*hr_bounds)
        gdf = gpd.GeoDataFrame(geometry=[bbox], crs=hr_crs)
        geom = [gdf.to_crs(src.crs).buffer(CLIP_BUFFER_M).geometry.values[0].__geo_interface__]
        clipped, clip_transform = mask(src, geom, crop=True)
        clip_meta = src.meta.copy()
        clip_meta.update({"height": clipped.shape[1], "width": clipped.shape[2],
                           "transform": clip_transform, "dtype": "float32"})
        tmp_clip = output_path.parent / f"_tmp_clip_{output_path.name}"
        with rasterio.open(tmp_clip, "w", **clip_meta) as tmp:
            tmp.write(clipped.astype(np.float32))
    with rasterio.open(tmp_clip) as src:
        dst_data = np.zeros((hr_height, hr_width), dtype=np.float32)
        reproject(source=rasterio.band(src, 1), destination=dst_data,
                   src_transform=src.transform, src_crs=src.crs,
                   dst_transform=hr_transform, dst_crs=hr_crs, resampling=resampling)
    out_meta = {"driver": "GTiff", "height": hr_height, "width": hr_width, "count": 1,
                "dtype": "float32", "crs": hr_crs, "transform": hr_transform}
    with rasterio.open(output_path, "w", **out_meta) as dst:
        dst.write(dst_data, 1)
    tmp_clip.unlink()


def extract_lr_tile(lr_data, row, col):
    region = lr_data[row:row+HR_TILE, col:col+HR_TILE]
    h, w = region.shape
    h4, w4 = (h // 4) * 4, (w // 4) * 4
    region = region[:h4, :w4]
    return region.reshape(h4//4, 4, w4//4, 4).mean(axis=(1, 3)).astype(np.float32)


def extract_qa_tile(qa_data, row, col):
    return qa_data[row:row+HR_TILE:4, col:col+HR_TILE:4]


def tile_is_valid(r, c, hr_data, lr_celsius, qa_data, hr_nodata):
    hr_tile = hr_data[r:r+HR_TILE, c:c+HR_TILE]
    lr_tile = extract_lr_tile(lr_celsius, r, c)
    qa_tile = extract_qa_tile(qa_data, r, c)
    hr_valid_mask = build_valid_mask(hr_tile, hr_nodata)
    if not (MIN_VALID_FRAC <= hr_valid_mask.mean() <= MAX_VALID_FRAC) or \
       (np.isfinite(lr_tile).mean() < MIN_VALID_FRAC):
        return False
    if extract_bit(qa_tile, BIT_CLOUD).mean() > MAX_CLOUD_FRAC or \
       extract_bit(qa_tile, BIT_CLOUD_SHADOW).mean() > MAX_SHADOW_FRAC:
        return False
    return True


def generate_candidate_patches(valid_mask, row_start, row_end, patch_step):
    patches = []
    height, width = valid_mask.shape
    for r in range(row_start, row_end - HR_TILE + 1, patch_step):
        for c in range(0, width - HR_TILE + 1, patch_step):
            footprint = valid_mask[r:r+HR_TILE, c:c+HR_TILE]
            if footprint.mean() >= PATCH_VALID_THRESHOLD:
                patches.append({"row_origin": r, "col_origin": c})
    return patches


def order_patches_north_south(patches):
    if len(patches) == 0:
        return []
    remaining = patches.copy()
    current = min(remaining, key=lambda p: (p["row_origin"], p["col_origin"]))
    remaining.remove(current)
    ordered = [current]
    MAX_SPATIAL_JUMP = HR_TILE * 2.5
    while remaining:
        candidates = []
        for idx, p in enumerate(remaining):
            dr = p["row_origin"] - current["row_origin"]
            dc = p["col_origin"] - current["col_origin"]
            distance = np.sqrt(dr**2 + dc**2)
            if distance > MAX_SPATIAL_JUMP:
                continue
            direction_penalty = 0 if dr >= 0 else HR_TILE * 3
            candidates.append((distance + direction_penalty, idx))
        if len(candidates) == 0:
            distances = []
            for idx, p in enumerate(remaining):
                dr = p["row_origin"] - current["row_origin"]
                dc = p["col_origin"] - current["col_origin"]
                distances.append((np.sqrt(dr**2 + dc**2), idx))
            _, idx = min(distances)
        else:
            _, idx = min(candidates)
        current = remaining.pop(idx)
        ordered.append(current)
    return ordered


def count_sequences(track_len, seq_step):
    """How many sequences a track of this length yields, given seq_step."""
    if track_len < SEQ_LEN:
        return 0
    return len(range(0, track_len - SEQ_LEN + 1, seq_step))


# ============================================================
# Main sweep
# ============================================================
def main():
    results = []  # rows: campaign, block, split, patch_step, track_len, n_sequences

    for hr_path in sorted(HR_DIR.glob("*.tif")):
        fname, campaign = hr_path.name, hr_path.stem
        if fname not in pair_lookup:
            print(f"No pair found for {fname}, skipping")
            continue
        pair_info = pair_lookup[fname]
        ls_tar = LS_DIR / pair_info["ls_name"]
        is_test = fname in TEST_CAMPAIGNS

        print(f"\n{'='*60}\n{fname} ({'TEST' if is_test else 'TRAIN/VAL'})")
        tmp_dir = Path(tempfile.mkdtemp())
        try:
            b10_path, qa_path, mtl_path = extract_scene_files(ls_tar, tmp_dir)
            mtl_data = parse_mtl_numeric(mtl_path)
            scale = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"]
            offset = mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]
            b10_aligned, qa_aligned = tmp_dir / "b10_aligned.tif", tmp_dir / "qa_aligned.tif"
            clip_to_hr(b10_path, hr_path, b10_aligned, Resampling.bilinear)
            clip_to_hr(qa_path, hr_path, qa_aligned, Resampling.nearest)

            with rasterio.open(hr_path) as src:
                hr_data = src.read(1).astype(np.float32)
                hr_nodata = src.nodata
            with rasterio.open(b10_aligned) as src:
                b10_data = src.read(1).astype(np.float32)
            with rasterio.open(qa_aligned) as src:
                qa_data = src.read(1)

            if hr_nodata is not None:
                hr_data[hr_data < -1e30] = np.nan
            b10_valid = (b10_data > 0)
            lr_celsius = np.where(b10_valid, b10_data * scale + offset - 273.15, np.nan).astype(np.float32)
            hr_valid_full = build_valid_mask(hr_data, hr_nodata)

            blocks = [{"block_id": 0, "row_start": 0, "row_end": hr_data.shape[0], "split": "test"}] \
                if is_test else split_map[fname]["blocks"]

            for block in blocks:
                split = block["split"]
                block_id = block["block_id"]
                row_start, row_end = block["row_start"], block["row_end"]

                step_candidates = TRAIN_STEP_CANDIDATES if split == "train" else EVAL_STEP_CANDIDATES
                seq_step = 1 if split == "train" else SEQ_LEN  # val fixed to SEQ_LEN per the bugfix

                for patch_step in step_candidates:
                    patch_locations = generate_candidate_patches(hr_valid_full, row_start, row_end, patch_step)
                    patch_locations = order_patches_north_south(patch_locations)

                    track_len = 0
                    for loc in patch_locations:
                        r, c = loc["row_origin"], loc["col_origin"]
                        if tile_is_valid(r, c, hr_data, lr_celsius, qa_data, hr_nodata):
                            track_len += 1

                    n_seq = count_sequences(track_len, seq_step)
                    results.append({
                        "campaign": campaign, "block": block_id, "split": split,
                        "patch_step": patch_step, "track_len": track_len,
                        "seq_step": seq_step, "n_sequences": n_seq,
                    })
                    print(f"  block {block_id} ({split}) step={patch_step}: "
                          f"track_len={track_len} -> {n_seq} sequences (seq_step={seq_step})")
        finally:
            shutil.rmtree(tmp_dir)

    # ------------------------------------------------------------
    # Split-level totals per patch_step candidate
    # ------------------------------------------------------------
    print("\n" + "=" * 60)
    print("TOTALS by split and patch_step")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        candidates = TRAIN_STEP_CANDIDATES if split == "train" else EVAL_STEP_CANDIDATES
        for step in candidates:
            total = sum(r["n_sequences"] for r in results if r["split"] == split and r["patch_step"] == step)
            n_blocks_zero = sum(1 for r in results if r["split"] == split and r["patch_step"] == step and r["n_sequences"] == 0)
            print(f"  {split:5s} step={step:3d}: total sequences = {total:4d}  "
                  f"(blocks with 0 sequences: {n_blocks_zero})")

    out_path = BASE / "seq_yield_report.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(results, f, indent=2)
    print(f"\nFull per-block results written to {out_path}")


if __name__ == "__main__":
    main()