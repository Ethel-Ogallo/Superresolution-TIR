"""
dataset.py — TIR Super-Resolution dataset and data utilities.

Provides:
  - compute_mean_std : train-only pixel statistics (no data leakage)
  - SRDataset        : returns (lr_t, hr_t, hr_mask) each (1, H, W)

Channel repetition (1→3) is handled by each model's forward().
The dataset stays model-agnostic.
"""

import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset


def compute_mean_std(metadata: pd.DataFrame) -> tuple[float, float]:
    """
    Compute mean and std over all valid LR pixels.
    Call on the training split only to avoid data leakage.
    Reads nodata value directly from the raster metadata.
    """
    pixel_sum = pixel_sq = pixel_count = 0.0

    for _, row in metadata.iterrows():
        with rasterio.open(row["lr_path"]) as src:
            nodata = src.nodata
            lr     = src.read(1).astype(np.float32)

        valid = lr[lr != nodata] if nodata is not None else lr[~np.isnan(lr)]
        if len(valid) == 0:
            continue

        pixel_sum   += valid.sum()
        pixel_sq    += (valid ** 2).sum()
        pixel_count += len(valid)

    mean = pixel_sum / pixel_count
    std  = np.sqrt(max(pixel_sq / pixel_count - mean ** 2, 0.0))
    print(f"  Train mean : {mean:.4f} °C  |  std : {std:.4f} °C")
    return float(mean), float(std)


class SRDataset(Dataset):
    """
    Single-channel TIR super-resolution dataset.

    Returns:
        lr_t   (1, H, W)   — normalised LR patch
        hr_t   (1, H*4, W*4) — normalised HR patch
        mask_t (1, H*4, W*4) — 1 = valid pixel, 0 = nodata

    Notes:
        - nodata value is read from the raster, not hardcoded
        - nodata pixels filled with mean before normalisation
        - mask is never normalised
        - channel repetition (1→3) is done inside each model's forward()
    """

    def __init__(self, metadata: pd.DataFrame,
                 mean: float = None, std: float = None):
        self.samples = metadata.reset_index(drop=True)
        self.mean    = mean
        self.std     = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        with rasterio.open(row["lr_path"]) as src:
            nodata_lr = src.nodata
            lr        = src.read(1).astype(np.float32)

        with rasterio.open(row["hr_path"]) as src:
            nodata_hr = src.nodata
            hr        = src.read(1).astype(np.float32)

        # Build mask BEFORE filling nodata
        if nodata_hr is not None:
            hr_mask = (hr != nodata_hr).astype(np.float32)
        else:
            hr_mask = (~np.isnan(hr)).astype(np.float32)

        # Fill nodata with a scalar (mean or 0)
        fill = float(self.mean) if self.mean is not None else 0.0

        if nodata_hr is not None:
            hr[hr == nodata_hr] = fill
        else:
            hr[np.isnan(hr)] = fill

        if nodata_lr is not None:
            lr[lr == nodata_lr] = 0.0
        else:
            lr[np.isnan(lr)] = 0.0

        # (H, W) → (1, H, W)
        lr_t   = torch.from_numpy(lr).unsqueeze(0).float()
        hr_t   = torch.from_numpy(hr).unsqueeze(0).float()
        mask_t = torch.from_numpy(hr_mask).unsqueeze(0).float()

        # Normalise inputs only — mask stays as 0/1
        if self.mean is not None and self.std is not None:
            lr_t = (lr_t - self.mean) / self.std
            hr_t = (hr_t - self.mean) / self.std

        return lr_t, hr_t, mask_t