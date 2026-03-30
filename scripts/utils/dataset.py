"""
dataset.py — TIR Super-Resolution dataset and data utilities.

Provides:
- Mean/std computation for LR pixels
- PyTorch Dataset class for single-channel TIR data, with automatic nodata handling
"""

import os
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset

# --------------- Compute mean and std from dataset ------------------
def compute_mean_std(metadata: pd.DataFrame) -> tuple[float, float]:
    """
    Compute mean and standard deviation over all valid LR pixels.
    Automatically handles raster nodata values.

    Args:
        metadata (pd.DataFrame): Metadata table of HR/LR pairs

    Returns:
        tuple[float, float]: Mean and standard deviation of valid LR pixels
    """
    pixel_sum = pixel_sq = pixel_count = 0.0

    for _, row in metadata.iterrows():
        with rasterio.open(row["lr_path"]) as src:
            nodata_val = src.nodata
            lr = src.read(1).astype(np.float32)

        if nodata_val is not None:
            valid = lr[lr != nodata_val]
        else:
            # fallback for rasters without explicit nodata
            valid = lr[~np.isnan(lr)]

        if len(valid) == 0:
            continue

        pixel_sum += valid.sum()
        pixel_sq += (valid ** 2).sum()
        pixel_count += len(valid)

    mean = pixel_sum / pixel_count
    variance = pixel_sq / pixel_count - mean ** 2
    variance = max(variance, 0.0)
    std = np.sqrt(variance)

    return float(mean), float(std)
# --------------------- PyTorch Dataset ----------------------
class SRDataset(Dataset):
    """
    PyTorch Dataset for single-channel TIR super-resolution.

    Returns:
        (lr_t, hr_t, hr_mask): Tensors of shape (3, H, W) after channel repetition.
    
    Features:
        - Automatically detects raster nodata values
        - Fills nodata pixels with mean
        - hr_mask: 1 = valid pixel, 0 = nodata (used for masked loss/metrics)
        - No augmentation applied (thermal data)
    """

    def __init__(self, metadata: pd.DataFrame, mean: float = None, std: float = None):
        """
        Args:
            metadata (pd.DataFrame): Metadata table of HR/LR pairs
            mean (float, optional): Mean for normalization
            std (float, optional): Std for normalization
        """
        self.samples = metadata.reset_index(drop=True)
        self.mean = mean
        self.std = std

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        """
        Load LR/HR pair and mask, repeat channels, normalize.
        Automatically handles raster nodata values.
        """
        row = self.samples.iloc[idx]

        # Load LR
        with rasterio.open(row["lr_path"]) as src:
            nodata_val_lr = src.nodata
            lr = src.read(1).astype(np.float32)

        # Load HR
        with rasterio.open(row["hr_path"]) as src:
            nodata_val_hr = src.nodata
            hr = src.read(1).astype(np.float32)

        # Build HR mask before filling nodata
        if nodata_val_hr is not None:
            hr_mask = (hr != nodata_val_hr).astype(np.float32)
        else:
            hr_mask = (~np.isnan(hr)).astype(np.float32)

        # Fill nodata
        if nodata_val_hr is not None:
            hr[hr == nodata_val_hr] = self.mean if self.mean is not None else 0.0
        else:
            hr[np.isnan(hr)] = self.mean if self.mean is not None else 0.0

        if nodata_val_lr is not None:
            lr[lr == nodata_val_lr] = 0.0
        else:
            lr[np.isnan(lr)] = 0.0

        # Convert to tensors, add channel dimension
        lr_t = torch.from_numpy(lr).unsqueeze(0).float()
        hr_t = torch.from_numpy(hr).unsqueeze(0).float()
        mask_t = torch.from_numpy(hr_mask).unsqueeze(0).float()

        # Repeat channels to match pretrained 3-channel models
        lr_t = lr_t.repeat(3, 1, 1)
        hr_t = hr_t.repeat(3, 1, 1)
        mask_t = mask_t.repeat(3, 1, 1)

        # Normalize if mean/std provided
        if self.mean is not None and self.std is not None:
            lr_t = (lr_t - self.mean) / self.std
            hr_t = (hr_t - self.mean) / self.std

        return lr_t, hr_t, mask_t