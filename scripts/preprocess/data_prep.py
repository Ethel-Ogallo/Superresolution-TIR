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
HR_PATCH_SIZE = 256  #256 or 512
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR
HR_STRIDE = 128 #HR_PATCH_SIZE // 2
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


def build_valid_mask(arr, nodata):
    """Return mask of valid (finite and non-nodata) pixels."""
    mask = np.isfinite(arr)
    if nodata is None:
        return mask
    if np.isnan(nodata):
        return mask & ~np.isnan(arr)

    atol = max(1e-6, abs(float(nodata)) * 1e-6)
    return mask & ~np.isclose(arr, nodata, rtol=0.0, atol=atol)

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

def clip_and_downsample_hr(hr_path, out_path, t_max=60):
    """Clip temperatures and downsample HR to 7.5m resolution ."""
    gt_res = LR_RESOLUTION_M / SCALE_FACTOR
    with rasterio.open(hr_path) as src:
        data = src.read(1).astype(np.float32)
        nodata = src.nodata

        # clip before downsampling so artifacts don't contaminate neighbors
        valid_mask = (data != nodata) if nodata is not None else np.ones(data.shape, bool)
        data[valid_mask] = np.clip(data[valid_mask], a_min=None, a_max=t_max)

        transform = from_origin(src.transform.c, src.transform.f, gt_res, gt_res)
        width = int(src.width * src.res[0] / gt_res)
        height = int(src.height * src.res[1] / gt_res)
        meta = src.meta.copy()
        meta.update({"width": width, "height": height, "transform": transform, "dtype": "float32"})

        with rasterio.MemoryFile() as memfile:
            with memfile.open(**src.meta) as mem:
                mem.write(data, 1)
            with memfile.open() as mem:
                with rasterio.open(out_path, "w", **meta) as dst:
                    reproject(
                        source=rasterio.band(mem, 1),
                        destination=rasterio.band(dst, 1),
                        src_transform=src.transform,
                        src_crs=src.crs,
                        dst_transform=transform,
                        dst_crs=src.crs,
                        resampling=Resampling.cubic
                    )

# ---------------- PATCH LOGIC ----------------
def compute_positions(
    hr_full,
    hr_nodata,
    hr_patch_size,
    hr_stride,
    min_valid_frac=0.10,
    max_valid_frac=0.95,
    min_patches=20,
    max_patches=2000,
    row_bins=24,
):
    """Compute HR patch origins from valid-pixel coverage only.

    This keeps tiling simple for single-band TIR strips and still enforces
    meaningful river presence in each patch. If filtering is too strict for a
    scene, thresholds are relaxed automatically.
    """
    hr_h, hr_w = hr_full.shape

    valid_mask = build_valid_mask(hr_full, hr_nodata)

    rows, cols = np.where(valid_mask)
    if len(rows) == 0:
        return []

    r_min, r_max = int(rows.min()), int(rows.max())
    c_min, c_max = int(cols.min()), int(cols.max())

    # Build row/col starts within valid bbox and include trailing edge coverage.
    row_starts = list(range(r_min, max(r_min + 1, r_max - hr_patch_size + 1), hr_stride))
    col_starts = list(range(c_min, max(c_min + 1, c_max - hr_patch_size + 1), hr_stride))

    row_last = max(0, min(r_max - hr_patch_size + 1, hr_h - hr_patch_size))
    col_last = max(0, min(c_max - hr_patch_size + 1, hr_w - hr_patch_size))
    row_starts.append(row_last)
    col_starts.append(col_last)

    row_starts = sorted(set(row_starts))
    col_starts = sorted(set(col_starts))

    candidates = []
    for row in row_starts:
        for col in col_starts:
            r0 = max(0, min(row, hr_h - hr_patch_size))
            c0 = max(0, min(col, hr_w - hr_patch_size))
            patch_mask = valid_mask[r0:r0 + hr_patch_size, c0:c0 + hr_patch_size]
            valid_frac = float(patch_mask.mean())
            if min_valid_frac <= valid_frac <= max_valid_frac:
                candidates.append((r0, c0, valid_frac))

    # If a scene is too narrow/complex, relax filters so it still contributes.
    if len(candidates) < min_patches:
        relaxed_min = max(0.03, min_valid_frac * 0.5)
        candidates = []
        for row in row_starts:
            for col in col_starts:
                r0 = max(0, min(row, hr_h - hr_patch_size))
                c0 = max(0, min(col, hr_w - hr_patch_size))
                patch_mask = valid_mask[r0:r0 + hr_patch_size, c0:c0 + hr_patch_size]
                valid_frac = float(patch_mask.mean())
                if valid_frac >= relaxed_min:
                    candidates.append((r0, c0, valid_frac))

    # Keep positions representative along the river length and coverage levels.
    # This avoids over-sampling one dense zone while still keeping edge/core structure.
    unique_candidates = sorted({(r0, c0, vf) for r0, c0, vf in candidates})
    if not unique_candidates:
        print("  Candidate patches after valid coverage filter: 0")
        return []

    if len(unique_candidates) <= max_patches:
        positions = [(r0, c0) for r0, c0, _ in unique_candidates]
    else:
        r_span = max(1, (r_max - r_min + 1))
        n_bins = max(1, row_bins)
        per_bin_cap = max(1, int(np.ceil(max_patches / n_bins)))

        selected = []
        for b in range(n_bins):
            b0 = r_min + (b * r_span) // n_bins
            b1 = r_min + ((b + 1) * r_span) // n_bins
            in_bin = [c for c in unique_candidates if b0 <= c[0] < b1]
            if not in_bin:
                continue

            edge = [c for c in in_bin if 0.10 <= c[2] < 0.25]
            mid = [c for c in in_bin if 0.25 <= c[2] < 0.60]
            core = [c for c in in_bin if 0.60 <= c[2] <= 0.95]

            quota_edge = max(1, per_bin_cap // 4)
            quota_mid = max(1, per_bin_cap // 2)
            quota_core = max(1, per_bin_cap - quota_edge - quota_mid)

            selected.extend(edge[:quota_edge])
            selected.extend(mid[:quota_mid])
            selected.extend(core[:quota_core])

            if len(selected) >= max_patches:
                break

        if len(selected) < max_patches:
            selected_set = set(selected)
            for c in unique_candidates:
                if c in selected_set:
                    continue
                selected.append(c)
                if len(selected) >= max_patches:
                    break

        positions = sorted({(r0, c0) for r0, c0, _ in selected})

    print(f"  Candidate patches after valid coverage filter: {len(positions)}")
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
            # clip_hr_temperatures(hr_path, paths["hr_clipped"])
            downsample_hr(hr_path, paths["hr_gt"])

            full_hr_path = os.path.join(full_hr_dir, f"{hr_id}.tif")
            if not os.path.exists(full_hr_path):
                shutil.copy(paths["hr_gt"], full_hr_path)

            # Scale/offset convert Landsat ST_B10 values to Celsius.
            mtl_data = parse_mtl_numeric(mtl)
            scale = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"]
            offset = mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]

            with rasterio.open(paths["l8_align"]) as lr_src, \
                rasterio.open(paths["hr_gt"]) as hr_src, \
                rasterio.open(paths["qa_align"]) as qa_src:

                lr = lr_src.read(1)
                qa_arr = qa_src.read(1)

                lr_t = lr_src.transform
                hr_t = hr_src.transform
                crs = hr_src.crs
                hr_nodata = hr_src.nodata
                hr_full = hr_src.read(1)

                positions = compute_positions(hr_full, hr_nodata, HR_PATCH_SIZE, HR_STRIDE)

                print(f"  Total candidate patches: {len(positions)}")

                metadata = []
                saved = 0
                skipped = 0
                skip_shape = 0
                skip_cloud = 0

                for patch_num, (hr_row, hr_col) in enumerate(positions, start=1):

                    hr_patch = pad_patch(
                        hr_full[
                            hr_row:hr_row + HR_PATCH_SIZE,
                            hr_col:hr_col + HR_PATCH_SIZE
                        ]
                    )

                    # skip empty patches
                    if not np.any(build_valid_mask(hr_patch, hr_nodata)):
                        skipped += 1
                        continue

                    # map HR → LR
                    x_min = hr_t.c + hr_col * hr_t.a
                    y_max = hr_t.f + hr_row * hr_t.e
                    x_max = x_min + HR_PATCH_SIZE * hr_t.a
                    y_min = y_max + HR_PATCH_SIZE * hr_t.e

                    lr_window = rasterio.windows.from_bounds(
                        x_min, y_min, x_max, y_max, transform=lr_t
                    )
                    lr_window = lr_window.round_offsets().round_lengths()

                    lr_row = int(lr_window.row_off)
                    lr_col = int(lr_window.col_off)

                    lr_patch = lr[
                        lr_row:lr_row + LR_PATCH_SIZE,
                        lr_col:lr_col + LR_PATCH_SIZE
                    ]

                    qa_patch = qa_arr[
                        lr_row:lr_row + LR_PATCH_SIZE,
                        lr_col:lr_col + LR_PATCH_SIZE

                    if lr_patch.shape != (LR_PATCH_SIZE, LR_PATCH_SIZE):
                        skipped += 1
                        skip_shape += 1
                        continue

                    cloud_only = extract_bit(qa_patch, BIT_CLOUD)
                    shadow_only = extract_bit(qa_patch, BIT_CLOUD_SHADOW)

                    if cloud_only.mean() > 0.40 or shadow_only.mean() > 0.10:
                        skipped += 1
                        skip_cloud += 1
                        continue

                    # convert to Celsius
                    lr_patch = lr_patch * scale + offset - 273.15

                    # transforms
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

        # print(f"\nScene: {hr_id} | Saved: {saved} | Skipped: {skipped} | Metadata: {meta_path}")
        print(f"\nScene: {hr_id} | Saved: {saved} | Skipped: {skipped} (shape:{skip_shape} cloud:{skip_cloud}) | Metadata: {meta_path}")
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
    parser.add_argument("--debug-scene", default=None, help="Run only on this HR filename (e.g. 'DZM_2013.tif') for testing")
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

    if args.debug_scene:
        scene_pairs = [(hr, ls) for hr, ls in scene_pairs
                    if os.path.basename(hr) == args.debug_scene]
        if not scene_pairs:
            raise ValueError(f"--debug-scene '{args.debug_scene}' not found in pairs")
        print(f"DEBUG MODE: running single scene {scene_pairs[0][0]}")
        # run directly, not in parallel
        process_scene(*scene_pairs[0], args.out_dir, patch_dir, metadata_dir)
        return

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
# python scripts/preprocess/prep_copy.py \
#     --hr-folder TIR_data/HR \
#     --ls-folder TIR_data/LR \
#     --out-dir TIR_data/processed \
#     --pairs-json TIR_data/processed/hr_lr_pairs.json \
#     --workers 2

# python scripts/preprocess/prep_copy.py \
#     --hr-folder TIR_data/HR \
#     --ls-folder TIR_data/LR \
#     --out-dir TIR_data/processed \
#     --pairs-json TIR_data/processed/hr_lr_pairs.json \
#     --debug-scene BAS-2025_TEMP_0.40_v11.tif