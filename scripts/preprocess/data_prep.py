import os
import glob
import json
import tarfile
import shutil
import argparse
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
HR_PATCH_SIZE = 512
LR_PATCH_SIZE = HR_PATCH_SIZE // SCALE_FACTOR
HR_STRIDE = HR_PATCH_SIZE // 2
CLIP_BUFFER_M = 300
MAX_CLOUD_FRAC = 0.10
BIT_CLOUD = 3
BIT_CLOUD_SHADOW = 4

# ---------------- UTILITIES ----------------
def read_nodata(path) -> float:
    with rasterio.open(path) as src:
        return src.nodata

def parse_mtl_numeric(path):
    """Return only numeric values from MTL file."""
    data = {}
    with open(path) as f:
        for line in f:
            if "=" in line:
                k, v = line.split("=")
                k = k.strip()
                v = v.strip().strip('"')
                try:
                    data[k] = float(v)
                except ValueError:
                    continue
    return data

def extract_bit(arr, bit):
    return ((arr.astype(np.uint16) >> bit) & 1) == 1

def get_hr_name(path):
    base = os.path.basename(path)
    parts = base.split("-")
    return f"{parts[0]}_{parts[1][:4]}"

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
    """Downsample HR raster → GT resolution."""
    gt_res = 30 / SCALE_FACTOR
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

def compute_positions(gt_path, hr_nodata) -> List[Tuple[int, int]]:
    with rasterio.open(gt_path) as src:
        hr_full = src.read(1).astype(np.float32)
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
def process_scene(hr_path, ls_tar, out_dir, patch_dir):
    try:
        hr_id = get_hr_name(hr_path)
        print(f"\nProcessing: {hr_id}")

        tmp = os.path.join(out_dir, f"_tmp_{hr_id}")
        os.makedirs(tmp, exist_ok=True)

        # Extract Landsat files
        with tarfile.open(ls_tar) as tar:
            tar.extractall(tmp)
        files = os.listdir(tmp)
        l8 = os.path.join(tmp, [f for f in files if "ST_B10" in f][0])
        qa = os.path.join(tmp, [f for f in files if "QA_PIXEL" in f][0])
        mtl = os.path.join(tmp, [f for f in files if "_MTL.txt" in f][0])

        # Align and downsample
        l8_clip = os.path.join(tmp, "l8_clip.tif")
        qa_clip = os.path.join(tmp, "qa_clip.tif")
        l8_align = os.path.join(tmp, "l8_align.tif")
        qa_align = os.path.join(tmp, "qa_align.tif")
        hr_gt = os.path.join(tmp, "hr_gt.tif")

        clip_to_hr(l8, hr_path, l8_clip)
        clip_to_hr(qa, hr_path, qa_clip)
        reproject_to_hr(l8_clip, hr_path, l8_align, Resampling.bilinear)
        reproject_to_hr(qa_clip, hr_path, qa_align, Resampling.nearest)
        downsample_hr(hr_path, hr_gt)

        # Load arrays
        with rasterio.open(l8_align) as lr_src, \
             rasterio.open(hr_gt) as hr_src, \
             rasterio.open(qa_align) as qa_src:
            lr, hr, qa = lr_src.read(1), hr_src.read(1), qa_src.read(1)
            lr_t, hr_t, crs = lr_src.transform, hr_src.transform, hr_src.crs

        hr_nodata = read_nodata(hr_gt)

        # Landsat scaling
        mtl_data = parse_mtl_numeric(mtl)
        scale = mtl_data["TEMPERATURE_MULT_BAND_ST_B10"]
        offset = mtl_data["TEMPERATURE_ADD_BAND_ST_B10"]

        # River-centered patch positions
        positions = compute_positions(hr_gt, hr_nodata)
        print(f"  Total candidate patches: {len(positions)}")

        # Output folders
        lr_out = os.path.join(patch_dir, "LR")
        hr_out = os.path.join(patch_dir, "HR")
        os.makedirs(lr_out, exist_ok=True)
        os.makedirs(hr_out, exist_ok=True)

        metadata = []
        saved, skipped = 0, 0

        for patch_num, (hr_row, hr_col) in enumerate(positions, start=1):
            hr_patch = pad_patch(hr[hr_row:hr_row+HR_PATCH_SIZE,
                                  hr_col:hr_col+HR_PATCH_SIZE])
            if hr_nodata is not None and (hr_patch == hr_nodata).all():
                skipped += 1
                continue

            hr_x = hr_t.c + hr_col * hr_t.a
            hr_y = hr_t.f + hr_row * hr_t.e
            lr_col = int(round((hr_x - lr_t.c) / lr_t.a))
            lr_row = int(round((hr_y - lr_t.f) / lr_t.e))
            lr_patch = lr[lr_row:lr_row+LR_PATCH_SIZE,
                          lr_col:lr_col+LR_PATCH_SIZE]
            qa_patch = qa[lr_row:lr_row+LR_PATCH_SIZE,
                          lr_col:lr_col+LR_PATCH_SIZE]

            cloud = extract_bit(qa_patch, BIT_CLOUD) | extract_bit(qa_patch, BIT_CLOUD_SHADOW)
            if cloud.mean() > MAX_CLOUD_FRAC:
                skipped += 1
                continue

            lr_patch = lr_patch * scale + offset - 273.15

            hr_patch_transform = from_origin(hr_t.c + hr_col*hr_t.a,
                                            hr_t.f + hr_row*hr_t.e,
                                            abs(hr_t.a), abs(hr_t.e))
            lr_patch_transform = from_origin(lr_t.c + lr_col*lr_t.a,
                                            lr_t.f + lr_row*lr_t.e,
                                            abs(lr_t.a), abs(lr_t.e))

            name = f"{hr_id}_{patch_num:04d}.tif"

            # Save patches
            for arr, path, transform, size in [
                (lr_patch, os.path.join(lr_out, name), lr_patch_transform, LR_PATCH_SIZE),
                (hr_patch, os.path.join(hr_out, name), hr_patch_transform, HR_PATCH_SIZE)
            ]:
                meta = dict(driver="GTiff", height=size, width=size, count=1,
                            dtype="float32", crs=crs, transform=transform, nodata=hr_nodata)
                with rasterio.open(path, "w", **meta) as dst:
                    dst.write(arr.astype(np.float32), 1)

            metadata.append({
                "hr_image_id": hr_id,
                "patch_name": name,
                "hr_row": int(hr_row),
                "hr_col": int(hr_col),
                "lr_row": int(lr_row),
                "lr_col": int(lr_col),
                "hr_transform": list(hr_patch_transform)[:6],
                "lr_transform": list(lr_patch_transform)[:6],
                "crs": str(crs)
            })
            saved += 1

        # Save metadata
        meta_path = os.path.join(out_dir, f"metadata_{hr_id}.json")
        with open(meta_path, "w") as f:
            json.dump(metadata, f, indent=2)

        print(f"\nScene: {hr_id} | Saved: {saved} | Skipped: {skipped} | Metadata: {meta_path}")
        shutil.rmtree(tmp)
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
    args = parser.parse_args()

    patch_dir = os.path.join(args.out_dir, "sample_tir")
    os.makedirs(patch_dir, exist_ok=True)

    hr_files = glob.glob(os.path.join(args.hr_folder, "*.tif"))
    ls_tars = glob.glob(os.path.join(args.ls_folder, "*.tar"))

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = [ex.submit(process_scene, hr, ls, args.out_dir, patch_dir)
                   for hr in hr_files for ls in ls_tars]
        for f in futures:
            f.result()

    print("\nAll scenes processed.")

if __name__ == "__main__":
    main()

# usage
# python scripts/preprocess/prep.py \
#     --hr-folder /home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR+LS_test/HR \
#     --ls-folder /home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR+LS_test/LR \
#     --out-dir /home/ogallo/Documents/CDE/MSC_thesis/Superresolution-TIR/TIR+LS_test/processed \
#     --workers 4