"""
scripts/utils/inference_pipeline.py
"""

import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm

from scripts.utils.dataset import SRDataset


# ─────────────────────────────────────────────
# Dataset Builder
# ─────────────────────────────────────────────
def _build_dataset(split: str, patches_dir: Path, stats_path: Path) -> SRDataset:
    aux_dir = str(patches_dir / split / "AUX")
    return SRDataset(
        split=split,
        patches_dir=str(patches_dir),
        stats_path=str(stats_path),
        use_aux=True,
        use_water_mask=True,
        aux_dir=aux_dir,
        transform=None,
    )


# ─────────────────────────────────────────────
# Model Forward Pass (NORMALIZED → PHYSICAL)
# ─────────────────────────────────────────────
def _run_single(model, sample, device, hr_std, hr_mean):
    batch = {
        k: sample[k].unsqueeze(0).to(device)
        for k in (
            "lr",
            "aux_lr",
            "aux_mid",
            "aux_hr",
            "lr_time",
            "time_gap_hours",
            "date_gap_days",
        )
    }

    with torch.no_grad():
        sr = model(batch)

    # Denormalize SR from network output
    sr = sr[:, 0] * hr_std + hr_mean
    return sr.squeeze().cpu().numpy()  # Returns shape: (H, W) e.g., (256, 256)


# ─────────────────────────────────────────────
# MAIN PIPELINE
# ─────────────────────────────────────────────
def run_inference_pipeline(
    model,
    splits,
    patches_dir,
    metadata_path,
    stats_path,
    output_csv,
    output_npz,
    device=None,
    min_hr_valid_frac=0.1,
):
    patches_dir = Path(patches_dir)
    output_csv = Path(output_csv)
    output_npz = Path(output_npz)

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = model.to(device).eval()

    # ── Load Normalization Stats ───────────
    with open(stats_path) as f:
        stats_data = json.load(f)

    hr_mean = stats_data["hr"]["mean"]
    hr_std = stats_data["hr"]["std"]

    # ── Load Spatial Metadata ──────────────
    with open(metadata_path) as f:
        metadata = json.load(f)

    rows = []
    store = {
        "patch_id": [],
        "campaign": [],
        "row": [],
        "col": [],
        "lr": [],
        "sr": [],
        "hr": [],
        "hr_mask": [],
        "water_mask": [],
    }

    for split in splits:
        print(f"\n── Processing Split: {split} ──")
        ds = _build_dataset(split, patches_dir, stats_path)

        for i in tqdm(range(len(ds)), desc=split):
            sample = ds[i]

            key = Path(sample["fname"]).stem
            meta = metadata.get(key, {})

            # Filter out low-quality/mostly empty frames early
            if meta.get("hr_valid_frac", 1.0) < min_hr_valid_frac:
                continue

            # ── Run Inference ───────────────────
            try:
                sr_np = _run_single(model, sample, device, hr_std, hr_mean)
            except Exception as e:
                print(f"[WARN] Failed inference for {key}: {e}")
                continue

            # ── Denormalize Arrays & Drop Channel Dims ──
            # .squeeze() ensures shapes change from (1, H, W) to (H, W)
            hr_np = sample["hr"].squeeze().cpu().numpy() * hr_std + hr_mean
            lr_np = sample["lr"].squeeze().cpu().numpy() * hr_std + hr_mean

            hr_mask_np = (sample["hr_mask"].squeeze().cpu().numpy() > 0.5).astype(np.uint8)
            water_mask_np = (sample["water_mask"].squeeze().cpu().numpy() > 0.5).astype(np.uint8)

            # Safety check for any invalid predictions
            sr_np = np.nan_to_num(sr_np, nan=0.0)

            # ── Append to Memory Storage ────────
            store["patch_id"].append(key)
            store["campaign"].append(meta.get("campaign", "unknown"))
            store["row"].append(meta.get("row_origin", 0))
            store["col"].append(meta.get("col_origin", 0))

            store["lr"].append(lr_np.astype(np.float32))
            store["sr"].append(sr_np.astype(np.float32))
            store["hr"].append(hr_np.astype(np.float32))
            store["hr_mask"].append(hr_mask_np)
            store["water_mask"].append(water_mask_np)

            # ── Build Light Metadata Index ──────
            bounds = meta.get("bounds_2154", {})
            rows.append({
                "patch_id": key,
                "campaign": meta.get("campaign", "unknown"),
                "row": meta.get("row_origin", 0),
                "col": meta.get("col_origin", 0),
                "hr_valid_frac": meta.get("hr_valid_frac", 0.0),
                "cloud_frac": meta.get("cloud_frac", 0.0),
                "time_gap_hours": meta.get("time_gap_hours", 0.0),
                "easting": (bounds.get("left", 0) + bounds.get("right", 0)) / 2,
                "northing": (bounds.get("top", 0) + bounds.get("bottom", 0)) / 2,
            })

    # ── Save Index CSV ────────────────────────
    df = pd.DataFrame(rows)
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_csv, index=False)
    print(f"\nSaved metadata registry → {output_csv}")

    # ── Save Compressed Arrays ────────────────
    output_npz.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output_npz,
        patch_id=np.array(store["patch_id"]),
        campaign=np.array(store["campaign"]),
        row=np.array(store["row"]),
        col=np.array(store["col"]),
        lr=np.array(store["lr"], dtype=np.float32),          # Shape: (N, 64, 64)
        sr=np.array(store["sr"], dtype=np.float32),          # Shape: (N, 256, 256)
        hr=np.array(store["hr"], dtype=np.float32),          # Shape: (N, 256, 256)
        hr_mask=np.array(store["hr_mask"], dtype=np.uint8),  # Shape: (N, 256, 256)
        water_mask=np.array(store["water_mask"], dtype=np.uint8), # Shape: (N, 256, 256)
    )
    print(f"Saved complete inference tensors → {output_npz}")

    return df