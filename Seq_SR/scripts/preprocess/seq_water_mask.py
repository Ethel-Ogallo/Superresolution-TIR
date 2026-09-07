import json
import numpy as np
import rasterio
from pathlib import Path
import h5py

# ============================================================
BASE = Path("/share/home/e2406751/Superresolution-TIR/data")
HR_DIR = BASE / "HR_downsampled"
WATER_MASK_DIR = BASE / "AUX" / "water_masks"
OUT_DIR = BASE / "processed/seq_patches"

HR_TILE = 256

WATER_H5_PATH = OUT_DIR / "water_masks.h5"
METADATA_PATH = OUT_DIR / "frame_metadata.json"

# ============================================================
def main():
    all_water = []

    with open(METADATA_PATH) as f:
        frame_metadata = json.load(f)

    print(f"Found {len(frame_metadata)} frames. Extracting water masks...")

    for hr_path in sorted(HR_DIR.glob("*.tif")):
        campaign = hr_path.stem
        print(f"  → {campaign}")

        # Load HR to get transform / shape
        with rasterio.open(hr_path) as src:
            hr_transform = src.transform
            hr_crs = src.crs
            hr_shape = src.shape

        # Load water mask (assuming it's already aligned)
        water_tif = WATER_MASK_DIR / f"{campaign}_water_mask.tif"
        if not water_tif.exists():
            print(f"    Warning: No water mask for {campaign}")
            continue

        with rasterio.open(water_tif) as src:
            water_mask = src.read(1).astype(bool)

        # Extract patches using metadata
        for frame_id, info in frame_metadata.items():
            if info.get("campaign") != campaign:
                continue

            r_orig = info["row_origin"]
            c_orig = info["col_origin"]

            water_patch = water_mask[r_orig:r_orig + HR_TILE, c_orig:c_orig + HR_TILE]

            if water_patch.shape == (HR_TILE, HR_TILE):
                all_water.append(water_patch.astype(np.uint8))
            else:
                print(f"    Shape mismatch for {frame_id}")

    # Save
    all_water = np.array(all_water, dtype=np.uint8)
    print(f"Writing {len(all_water)} water mask patches...")

    with h5py.File(WATER_H5_PATH, 'w') as hf:
        hf.create_dataset('frames', data=all_water, compression='gzip', compression_opts=4)

    print(f"Done! Saved {WATER_H5_PATH}")

if __name__ == "__main__":
    main()