"""AUX Extraction: Reproject Sentinel-2 once per site, then window-read per patch."""
import os
import json
import glob
import argparse
import tempfile
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_origin
import numpy as np

SITE_TO_AUX = {
    "PDR":  "TIR_data/sentinel2_aux/PDR_aux.tif",
    "DZM":  "TIR_data/sentinel2_aux/DZM_aux.tif",
    "BRC":  "TIR_data/sentinel2_aux/BRC_aux.tif",
    "BAS":  "TIR_data/sentinel2_aux/BAS_aux.tif",
    "HAUT": "TIR_data/sentinel2_aux/HAUT_aux.tif",
}

HR_RES_M = 7.5
HR_PATCH_SIZE = 256
AUX_PATCH_SIZE = HR_PATCH_SIZE


def reproject_sentinel(aux_path, ref_metadata, out_path):
    """Reproject full Sentinel scene to 7.5m in the HR CRS once."""
    # Use first patch entry to get CRS and bounds of the full scene
    first = ref_metadata[0]
    dst_crs = first['crs']

    # Get full spatial extent from all patches
    lefts  = [e['hr_transform'][2] for e in ref_metadata]
    tops   = [e['hr_transform'][5] for e in ref_metadata]
    rights = [e['hr_transform'][2] + HR_PATCH_SIZE * e['hr_transform'][0] for e in ref_metadata]
    bots   = [e['hr_transform'][5] + HR_PATCH_SIZE * e['hr_transform'][4] for e in ref_metadata]

    left, top, right, bottom = min(lefts), max(tops), max(rights), min(bots)

    transform = from_origin(left, top, HR_RES_M, HR_RES_M)
    width  = int(round((right - left)  / HR_RES_M))
    height = int(round((top - bottom)  / HR_RES_M))

    with rasterio.open(aux_path) as src:
        meta = src.meta.copy()
        meta.update({
            "crs": dst_crs,
            "transform": transform,
            "width": width,
            "height": height,
            "dtype": "float32",
            "compress": "lzw"
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
                    resampling=Resampling.bilinear
                )
    print(f"  Reprojected Sentinel to {width}x{height} @ {HR_RES_M}m")


def extract_aux_patches(metadata_path, patch_dir):
    with open(metadata_path, 'r') as f:
        metadata = json.load(f)
    if not metadata:
        return

    sample_id = metadata[0]['hr_image_id']
    site = sample_id.split('_')[0]
    aux_path = SITE_TO_AUX.get(site)

    if not aux_path or not os.path.exists(aux_path):
        print(f"Skipping {sample_id}: No AUX file found at {aux_path}")
        return

    aux_out_dir = os.path.join(patch_dir, "AUX")
    os.makedirs(aux_out_dir, exist_ok=True)
    print(f"\nProcessing AUX for {sample_id}...")

    # Reproject once into a temp file, then window-read per patch
    with tempfile.NamedTemporaryFile(suffix=".tif", delete=False) as tmp:
        tmp_path = tmp.name

    try:
        reproject_sentinel(aux_path, metadata, tmp_path)

        with rasterio.open(tmp_path) as src:
            for entry in metadata:
                patch_name = entry['patch_name']
                t = entry['hr_transform']

                left   = t[2]
                top    = t[5]
                right  = left + HR_PATCH_SIZE * t[0]
                bottom = top  + HR_PATCH_SIZE * t[4]

                window = rasterio.windows.from_bounds(
                    left, bottom, right, top,
                    transform=src.transform
                )

                # Simple window read — no reproject needed
                out_arr = src.read(
                    window=window,
                    out_shape=(src.count, AUX_PATCH_SIZE, AUX_PATCH_SIZE),
                    resampling=Resampling.bilinear
                )

                out_path = os.path.join(aux_out_dir, patch_name)
                meta = {
                    "driver": "GTiff",
                    "height": AUX_PATCH_SIZE,
                    "width": AUX_PATCH_SIZE,
                    "count": src.count,
                    "dtype": "float32",
                    "crs": entry['crs'],
                    "transform": from_origin(left, top, t[0], abs(t[4])),
                    "compress": "lzw"
                }
                with rasterio.open(out_path, "w", **meta) as dst:
                    dst.write(out_arr)

        print(f"  Saved {len(metadata)} AUX patches")

    finally:
        os.remove(tmp_path)  # cleanup temp reprojected file


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata-dir", required=True)
    parser.add_argument("--patch-dir", required=True)
    args = parser.parse_args()

    meta_files = glob.glob(os.path.join(args.metadata_dir, "metadata_*.json"))
    for meta_file in sorted(meta_files):
        extract_aux_patches(meta_file, args.patch_dir)


if __name__ == "__main__":
    main()


# python scripts/preprocess/extract_aux.py \
#     --metadata-dir TIR_data/processed/metadata \
#     --patch-dir TIR_data/processed/tir_patches
