"""
create_metadata.py — Scan HR/LR directories and create metadata CSV for TIR SR datasets.

Features:
- Scan HR/LR folders, check alignment
- Save CSV with patch_id, hr_path, lr_path, split
- Fully importable for multi-dataset workflows

Usage (CLI):
    python scripts/preprocess/create_metadata.py \
    --hr_dir data/processed/sample_tir/HR \
    --lr_dir data/processed/sample_tir/LR \
    --output_csv data/metadata.csv
"""

import os
import argparse
import pandas as pd
import rasterio


def create_metadata(hr_dir: str, lr_dir: str, output_csv: str = "metadata.csv") -> pd.DataFrame:
    """
    Scan HR/LR directories and generate metadata CSV.

    Args:
        hr_dir (str): Path to HR images
        lr_dir (str): Path to LR images
        output_csv (str): Path to save CSV

    Returns:
        pd.DataFrame: Metadata table
    """
    records = []
    hr_files = sorted(f for f in os.listdir(hr_dir) if f.endswith((".tif", ".tiff")))
    missing = mismatches = 0

    for f in hr_files:
        hr_path = os.path.abspath(os.path.join(hr_dir, f))
        lr_path = os.path.abspath(os.path.join(lr_dir, f))

        if not os.path.exists(lr_path):
            print(f"  WARNING: missing LR file for {f}")
            missing += 1
            continue

        # Check spatial alignment
        with rasterio.open(hr_path) as h, rasterio.open(lr_path) as l:
            if h.width != l.width * 4 or h.height != l.height * 4:
                print(f"  WARNING: spatial mismatch — {f}")
                mismatches += 1
                continue

        records.append({
            "patch_id": os.path.splitext(f)[0],
            "hr_path": hr_path,
            "lr_path": lr_path,
            "split": None,
        })

    df = pd.DataFrame(records)
    df.to_csv(output_csv, index=False)
    print(f"Valid pairs : {len(df)}")
    if missing: print(f"Missing LR  : {missing}")
    if mismatches: print(f"Mismatches  : {mismatches}")

    return df


def parse_args():
    p = argparse.ArgumentParser(description="Build metadata CSV with full paths for HR/LR image pairs.")
    p.add_argument("--hr_dir", required=True, help="Directory with HR .tif files")
    p.add_argument("--lr_dir", required=True, help="Directory with LR .tif files")
    p.add_argument("--output_csv", default="metadata.csv", help="output to save the CSV")
    return p.parse_args()


def main():
    args = parse_args()
    create_metadata(args.hr_dir, args.lr_dir, args.output_csv)


if __name__ == "__main__":
    main()