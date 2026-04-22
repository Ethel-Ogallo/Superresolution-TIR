"""TIR preprocessing: align rasters, extract HR/LR patches, and save metadata."""

import os
import glob
import json
import re
import tarfile
import shutil
import argparse
import tempfile
from concurrent.futures import ProcessPoolExecutor
from typing import List, Tuple


import numpy as np
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
from rasterio.mask import mask
from shapely.geometry import box
import geopandas as gpd


# ---------------- PARAMETERS ----------------
SCALE_FACTOR = 4
LR_RESOLUTION_M = 30
HR_PATCH_SIZE = 512
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR
HR_STRIDE = HR_PATCH_SIZE // 2
CLIP_BUFFER_M = 300
MAX_CLOUD_FRAC = 0.10
BIT_CLOUD = 3
BIT_CLOUD_SHADOW = 4

# ---------------- UTILITIES ----------------
def parse_mtl_numeric(path):
    """Read numeric fields from an MTL text file."""
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

def extract_bit(arr, bit):
    """Return boolean mask for a bit position in QA raster."""
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

# def get_hr_name(path):
#     parts = os.path.basename(path).replace(".tif", "").split("_")
#     return f"{parts[0]}_{parts[1]}"  # e.g. "PDR_2013"
def get_hr_name(path):
    base = os.path.basename(path).replace(".tif", "")
    # normalize separators, drop TEMP suffix
    base = base.replace("-", "_").upper()
    parts = base.split("_")
    # keep only SITE + YEAR e.g. DZM_2013
    site = parts[0]
    year = next((p for p in parts if re.match(r"(19|20)\d{2}$", p)), "UNKNOWN")
    return f"{site}_{year}"

def clip_to_hr(input_path, hr_path, output_path):
    """Clip raster to HR footprint + buffer."""
    with rasterio.open(hr_path) as ref:
        bbox = box(*ref.bounds)
        gdf = gpd.GeoDataFrame(geometry=[bbox], crs=ref.crs)

    with rasterio.open(input_path) as src:
        geom = [gdf.to_crs(src.crs).buffer(CLIP_BUFFER_M).geometry.values[0].__geo_interface__]
        out, transform = mask(src, geom, crop=True)
        meta = src.meta.copy()
        meta.update({
            "height": out.shape[1],
            "width": out.shape[2],
            "transform": transform
        })

    with rasterio.open(output_path, "w", **meta) as dst:
        dst.write(out)

def reproject_to_hr(src_path, hr_path, out_path, resampling):
    """Reproject raster to HR CRS."""
    with rasterio.open(hr_path) as ref:
        dst_crs = ref.crs
        bounds = ref.bounds

    with rasterio.open(src_path) as src:
        res = src.res[0]
        transform = from_origin(bounds.left, bounds.top, res, res)
        width = int((bounds.right - bounds.left) / res)
        height = int((bounds.top - bounds.bottom) / res)

        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32"
        })

        with rasterio.open(out_path, "w", **meta) as dst:
            for i in range(1, src.count + 1):
                reproject(
                    source=rasterio.band(src, i),
                    destination=rasterio.band(dst, i),
                    src_transform=src.transform,
                    src_crs=src.crs,
                    dst_transform=transform,
                    dst_crs=dst_crs,
                    resampling=resampling
                )

def downsample_hr(hr_path, out_path):
    """Downsample HR raster to GT resolution."""
    gt_res = LR_RESOLUTION_M / SCALE_FACTOR
    with rasterio.open(hr_path) as src:
        transform = from_origin(src.transform.c, src.transform.f, gt_res, gt_res)
        width = int(src.width * src.res[0] / gt_res)
        height = int(src.height * src.res[1] / gt_res)
        meta = src.meta.copy()
        meta.update({"width": width, "height": height, "transform": transform})

        with rasterio.open(out_path, "w", **meta) as dst:
            reproject(
                source=rasterio.band(src, 1),
                destination=rasterio.band(dst, 1),
                src_transform=src.transform,
                src_crs=src.crs,
                dst_transform=transform,
                dst_crs=src.crs,
                resampling=Resampling.cubic
            )

# ---------------- PATCH LOGIC ----------------
def get_river_col(valid_mask, hr_row, hr_h, hr_w, patch_size):
    row_end = min(hr_row + patch_size, hr_h)
    row_band = valid_mask[hr_row:row_end, :]
    col_counts = row_band.sum(axis=0)
    valid_cols = np.where(col_counts > 0)[0]
    if len(valid_cols) == 0:
        return None
    weights = col_counts[valid_cols]
    river_col = int(np.average(valid_cols, weights=weights))
    hr_col = river_col - patch_size // 2
    return max(0, min(hr_col, hr_w - patch_size))

def compute_positions(hr_full, hr_nodata) -> List[Tuple[int, int]]:
    """Compute strip-aware HR patch origins from an HR array."""
    hr_full = hr_full.astype(np.float32)
    hr_h, hr_w = hr_full.shape
    valid_mask = (hr_full != hr_nodata) if hr_nodata is not None else (hr_full < 1e38)
    positions = []
    for hr_row in range(0, hr_h - HR_PATCH_SIZE + 1, HR_STRIDE):
        hr_col = get_river_col(valid_mask, hr_row, hr_h, hr_w, HR_PATCH_SIZE)
        if hr_col is not None:
            positions.append((hr_row, hr_col))
    # include last row if missing
    last_row = hr_h - HR_PATCH_SIZE
    if positions and positions[-1][0] != last_row:
        hr_col = get_river_col(valid_mask, last_row, hr_h, hr_w, HR_PATCH_SIZE)
        if hr_col is not None:
            positions.append((last_row, hr_col))
    return positions

def pad_patch(arr, target_size=HR_PATCH_SIZE):
    h, w = arr.shape
    if h == target_size and w == target_size:
        return arr
    padded = np.zeros((target_size, target_size), dtype=arr.dtype)
    padded[:h, :w] = arr
    return padded

# ---------------- PROCESS SCENE ----------------
def find_scene_files(extract_dir):
    """Locate required Landsat files after TAR extraction."""
    files = os.listdir(extract_dir)
    l8 = [f for f in files if "ST_B10" in f]
    qa = [f for f in files if "QA_PIXEL" in f]
    mtl = [f for f in files if "_MTL.txt" in f]
    if not l8 or not qa or not mtl:
        raise FileNotFoundError(f"Missing required scene files in {extract_dir}")
    return (
        os.path.join(extract_dir, l8[0]),
        os.path.join(extract_dir, qa[0]),
        os.path.join(extract_dir, mtl[0]),
    )

def build_temp_paths(tmp_dir):
    """Standard intermediate file paths for one scene."""
    names = ("l8_clip", "qa_clip", "l8_align", "qa_align", "hr_gt")
    return {name: os.path.join(tmp_dir, f"{name}.tif") for name in names}

def save_patch(path, arr, transform, size, crs, nodata):
    """Write one patch to disk with consistent GeoTIFF metadata."""
    meta = {
        "driver": "GTiff",
        "height": size,
        "width": size,
        "count": 1,
        "dtype": "float32",
        "crs": crs,
        "transform": transform,
        "nodata": nodata,
    }
    with rasterio.open(path, "w", **meta) as dst:
        dst.write(arr.astype(np.float32), 1)

def validate_scene_lists(hr_files, ls_tars, hr_folder, ls_folder):
    """Validate input file lists before launching parallel workers."""
    if not hr_files:
        raise FileNotFoundError(f"No HR .tif files found in {hr_folder}")
    if not ls_tars:
        raise FileNotFoundError(f"No Landsat .tar files found in {ls_folder}")


def extract_date_token(path):
    """Extract YYYYMMDD token from a filename, if present."""
    m = re.search(r"(19|20)\d{6}", os.path.basename(path))
    return m.group(0) if m else None


def build_scene_pairs(hr_files, ls_tars, pairs_json=None):
    """Build HR/Landsat pairs from manifest JSON (selected pairs only) or auto-matching."""
    hr_files = sorted(hr_files)
    ls_tars = sorted(ls_tars)

    if pairs_json:
        with open(pairs_json) as f:
            manifest = json.load(f)

        if isinstance(manifest, dict) and "pairs" in manifest:
            raw_pairs = manifest["pairs"]
            hr_by_name = {os.path.basename(p): p for p in hr_files}
            ls_by_name = {os.path.basename(p): p for p in ls_tars}
            resolved = []

            for entry in raw_pairs:
                hr_key = entry["hr_name"]
                candidates = entry.get("candidates", [])
                if not candidates:
                    continue

                selected_candidates = [c for c in candidates if c.get("selected", False)]
                if len(selected_candidates) > 1:
                    raise ValueError(
                        f"Entry '{hr_key}' has multiple selected candidates. "
                        "Please keep only one candidate with 'selected': true."
                    )

                # If only one candidate exists, accept it even without explicit selected=true.
                if len(selected_candidates) == 1:
                    ls_key = selected_candidates[0]["ls_name"]
                elif len(candidates) == 1:
                    ls_key = candidates[0]["ls_name"]
                else:
                    raise ValueError(
                        f"Entry '{hr_key}' has multiple candidates but none selected. "
                        "Mark one candidate with 'selected': true."
                    )

                hr_resolved = hr_by_name.get(os.path.basename(hr_key))
                ls_resolved = ls_by_name.get(os.path.basename(ls_key))
                if not hr_resolved:
                    raise FileNotFoundError(f"HR file not found in folder: {hr_key}")
                if not ls_resolved:
                    raise FileNotFoundError(f"Landsat file not found in folder: {ls_key}")
                resolved.append((hr_resolved, ls_resolved))

            if not resolved:
                raise ValueError(
                    f"No valid HR/Landsat pairs resolved from {pairs_json}. "
                    "Check 'hr_name', candidate 'ls_name', and selected flags."
                )
            return resolved
        elif isinstance(manifest, dict):
            pairs = list(manifest.items())
            hr_by_name = {os.path.basename(p): p for p in hr_files}
            ls_by_name = {os.path.basename(p): p for p in ls_tars}
            resolved = []
            for hr_key, ls_key in pairs:
                hr_resolved = hr_by_name.get(os.path.basename(hr_key))
                ls_resolved = ls_by_name.get(os.path.basename(ls_key))
                if not hr_resolved:
                    raise FileNotFoundError(f"HR file from mapping not found in folder: {hr_key}")
                if not ls_resolved:
                    raise FileNotFoundError(f"Landsat file from mapping not found in folder: {ls_key}")
                resolved.append((hr_resolved, ls_resolved))
            return resolved
        else:
            raise ValueError("pairs JSON must be either a mapping dict or {'pairs': [...]} format")

    # Automatic matching by shared date token when filenames differ.
    ls_by_date = {}
    for ls in ls_tars:
        d = extract_date_token(ls)
        if d:
            ls_by_date.setdefault(d, []).append(ls)

    auto_pairs = []
    for hr in hr_files:
        d = extract_date_token(hr)
        if d and d in ls_by_date and len(ls_by_date[d]) == 1:
            auto_pairs.append((hr, ls_by_date[d][0]))

    if len(auto_pairs) == len(hr_files):
        return auto_pairs

    # Final fallback: strict sorted zip if counts match.
    if len(hr_files) != len(ls_tars):
        raise ValueError(
            "Could not build robust pairs automatically. Provide --pairs-json mapping. "
            f"Found {len(hr_files)} HR files and {len(ls_tars)} Landsat TAR files."
        )
    return list(zip(hr_files, ls_tars))

def process_scene(hr_path, ls_tar, out_dir, patch_dir, metadata_dir):
    try:
        hr_id = get_hr_name(hr_path)
        print(f"\nProcessing: {hr_id}")

        # Output directories are created once here for process safety.
        lr_out = os.path.join(patch_dir, "LR")
        hr_out = os.path.join(patch_dir, "HR")
        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)

        full_hr_dir = os.path.join(out_dir, "HR_downsampled")
        os.makedirs(full_hr_dir, exist_ok=True)

        # Keep all scene intermediates in one temp folder so cleanup is automatic.
        with tempfile.TemporaryDirectory(prefix=f"tmp_{hr_id}_", dir=out_dir) as tmp:
            with tarfile.open(ls_tar) as tar:
                tar.extractall(tmp)
            l8, qa, mtl = find_scene_files(tmp)

            paths = build_temp_paths(tmp)
            clip_to_hr(l8, hr_path, paths["l8_clip"])
            clip_to_hr(qa, hr_path, paths["qa_clip"])
            reproject_to_hr(paths["l8_clip"], hr_path, paths["l8_align"], Resampling.bilinear)
            reproject_to_hr(paths["qa_clip"], hr_path, paths["qa_align"], Resampling.nearest)
            downsample_hr(hr_path, paths["hr_gt"])

            full_hr_path = os.path.join(full_hr_dir, f"{hr_id}.tif")
            if not os.path.exists(full_hr_path):
                shutil.copy(paths["hr_gt"], full_hr_path)

            # Scale/offset convert Landsat ST_B10 values to Celsius.
            mtl_data = parse_mtl_numeric(mtl)
            scale = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"]
            offset = mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]

            with rasterio.open(paths["l8_align"]) as lr_src, rasterio.open(paths["hr_gt"]) as hr_src, rasterio.open(paths["qa_align"]) as qa_src:
                lr = lr_src.read(1)
                hr = hr_src.read(1)
                qa_arr = qa_src.read(1)
                lr_t = lr_src.transform
                hr_t = hr_src.transform
                crs = hr_src.crs
                hr_nodata = hr_src.nodata

                positions = compute_positions(hr, hr_nodata)
                print(f"  Total candidate patches: {len(positions)}")

                metadata = []
                saved = 0
                skipped = 0

                # Build aligned patch pairs and filter unusable patches.
                for patch_num, (hr_row, hr_col) in enumerate(positions, start=1):
                    hr_patch = pad_patch(
                        hr[hr_row:hr_row + HR_PATCH_SIZE, hr_col:hr_col + HR_PATCH_SIZE]
                    )
                    if hr_nodata is not None and (hr_patch == hr_nodata).all():
                        skipped += 1
                        continue

                    # Convert HR patch bounds to LR indices to keep exact alignment.
                    x_min = hr_t.c + hr_col * hr_t.a
                    y_max = hr_t.f + hr_row * hr_t.e
                    x_max = x_min + HR_PATCH_SIZE * hr_t.a
                    y_min = y_max + HR_PATCH_SIZE * hr_t.e

                    lr_window = rasterio.windows.from_bounds(x_min, y_min, x_max, y_max, transform=lr_t)
                    lr_window = lr_window.round_offsets().round_lengths()

                    lr_row = int(lr_window.row_off)
                    lr_col = int(lr_window.col_off)
                    lr_patch = lr[lr_row:lr_row + LR_PATCH_SIZE, lr_col:lr_col + LR_PATCH_SIZE]
                    qa_patch = qa_arr[lr_row:lr_row + LR_PATCH_SIZE, lr_col:lr_col + LR_PATCH_SIZE]

                    if lr_patch.shape != (LR_PATCH_SIZE, LR_PATCH_SIZE):
                        skipped += 1
                        continue

                    cloud_only = extract_bit(qa_patch, BIT_CLOUD)
                    shadow_only = extract_bit(qa_patch, BIT_CLOUD_SHADOW)

                    if cloud_only.mean() > 0.40 or shadow_only.mean() > 0.10:
                        skipped += 1
                        continue

                    lr_patch = lr_patch * scale + offset - 273.15  # Convert to Celsius

                    hr_patch_transform = from_origin(
                        hr_t.c + hr_col * hr_t.a,
                        hr_t.f + hr_row * hr_t.e,
                        abs(hr_t.a),
                        abs(hr_t.e),
                    )
                    lr_patch_transform = lr_src.window_transform(lr_window)
                    name = f"{hr_id}_{patch_num}.tif"

                    save_patch(
                        os.path.join(lr_out, name),
                        lr_patch,
                        lr_patch_transform,
                        LR_PATCH_SIZE,
                        crs,
                        hr_nodata,
                    )
                    save_patch(
                        os.path.join(hr_out, name),
                        hr_patch,
                        hr_patch_transform,
                        HR_PATCH_SIZE,
                        crs,
                        hr_nodata,
                    )

                    metadata.append({
                        "hr_image_id": hr_id,
                        "patch_name": name,
                        "hr_row": int(hr_row),
                        "hr_col": int(hr_col),
                        "lr_row": int(lr_row),
                        "lr_col": int(lr_col),
                        "hr_transform": list(hr_patch_transform)[:6],
                        "lr_transform": list(lr_patch_transform)[:6],
                        "crs": str(crs),
                    })
                    saved += 1

            # Save one metadata file per scene for easier debugging and merges.
            meta_path = os.path.join(metadata_dir, f"metadata_{hr_id}.json")
            with open(meta_path, "w") as f:
                json.dump(metadata, f, indent=2)

        print(f"\nScene: {hr_id} | Saved: {saved} | Skipped: {skipped} | Metadata: {meta_path}")
        return saved

    except Exception as e:
        print(f"Error processing scene {hr_path}: {e}")
        return 0

# ---------------- MAIN ----------------
def main():
    parser = argparse.ArgumentParser(description="River-centered patch extraction")
    parser.add_argument("--hr-folder", required=True)
    parser.add_argument("--ls-folder", required=True)
    parser.add_argument("--out-dir", required=True)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--pairs-json",default=None,
        help=(
            "Optional HR-to-Landsat mapping JSON. "
            "Supported formats: {'hr_name.tif': 'ls_name.tar'} or {'pairs': [{'hr_name': '...', 'candidates': [...]}]}"
        ),
    )
    args = parser.parse_args()

    patch_dir = os.path.join(args.out_dir, "tir_patches")
    os.makedirs(patch_dir, exist_ok=True)

    metadata_dir = os.path.join(args.out_dir, "metadata")
    os.makedirs(metadata_dir, exist_ok=True)

    hr_files = sorted(glob.glob(os.path.join(args.hr_folder, "*.tif")))
    ls_tars = sorted(glob.glob(os.path.join(args.ls_folder, "*.tar")))

    validate_scene_lists(hr_files, ls_tars, args.hr_folder, args.ls_folder)

    scene_pairs = build_scene_pairs(hr_files, ls_tars, args.pairs_json)
    print(f"Processing {len(scene_pairs)} scene pairs...")

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [
            ex.submit(process_scene, hr, ls, args.out_dir, patch_dir, metadata_dir)
            for hr, ls in scene_pairs
        ]
        for f in futures:
            f.result()

    print("\nAll scenes processed.")

if __name__ == "__main__":
    main()

# usage
# python scripts/preprocess/prep.py \
#     --hr-folder TIR_data/HR \
#     --ls-folder TIR_data/LR \
#     --out-dir TIR_data/processed \
#     --pairs-json TIR_data/processed/hr_lr_pairs.json \
#     --workers 4

