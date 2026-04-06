"""
dataset.py — TIR Super-Resolution dataset and data utilities.
"""

import random
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# -------------- Data stats ---------------
def compute_mean_std(metadata: pd.DataFrame) -> tuple[float, float]:
    pixel_sum = pixel_sq = pixel_count = 0.0
    for _, row in metadata.iterrows():
        with rasterio.open(row["lr_path"]) as src:
            nodata = src.nodata
            lr     = src.read(1).astype(np.float32)

        valid = lr[lr != nodata] if nodata is not None else lr[~np.isnan(lr)]
        if len(valid) == 0: continue

        pixel_sum   += valid.sum()
        pixel_sq    += (valid ** 2).sum()
        pixel_count += len(valid)

    mean = pixel_sum / pixel_count
    std  = np.sqrt(max(pixel_sq / pixel_count - mean ** 2, 0.0))
    print(f"  Train mean : {mean:.4f} °C  |  std : {std:.4f} °C")
    return float(mean), float(std)

# ------------ Data Augmentations ----------------
class GeoAugment:
    """Geometric transforms: flips + 90° rotations."""

    def __call__(self, lr, hr, mask):
        if random.random() > 0.5:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
            mask = np.fliplr(mask).copy()
        if random.random() > 0.5:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
            mask = np.flipud(mask).copy()
        k = random.randint(0, 3)
        if k > 0:
            lr = np.rot90(lr, k).copy()
            hr = np.rot90(hr, k).copy()
            mask = np.rot90(mask, k).copy()
        return lr, hr, mask


class TIRNoise:
    """
    Adds calibrated noise to LR only.
    std should be estimated from actual LR (Landsat-8) patches.
    """

    def __init__(self, std, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, lr, hr, mask):
        if random.random() < self.p:
            multiplier = random.uniform(0.5, 1.5)
            noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
            lr = lr + noise
        return lr, hr, mask


class ThermalShift:
    """
    Global diurnal shift applied to both (physically consistent).
    Optional inter-sensor bias applied to LR only.
    """

    def __init__(self, range_c=2.0, sensor_bias_range=0.5):
        self.range_c = range_c
        self.sensor_bias = sensor_bias_range

    def __call__(self, lr, hr, mask):
        shift = random.uniform(-self.range_c, self.range_c)
        bias = random.uniform(-self.sensor_bias, self.sensor_bias)
        return lr + shift + bias, hr + shift, mask


class ContrastScaling:
    """
    Scales thermal gradients around each image's own mean.
    """

    def __init__(self, range_alpha=(0.90, 1.10)):
        self.range_alpha = range_alpha

    def __call__(self, lr, hr, mask):
        if random.random() > 0.5:
            alpha = random.uniform(*self.range_alpha)
            lr_m = lr.mean()
            hr_m = hr.mean()
            lr = (lr - lr_m) * alpha + lr_m
            hr = (hr - hr_m) * alpha + hr_m
        return lr, hr, mask


class BlurAugment:
    """
    Blurs LR only to simulate sensor PSF variability.
    Handles both (H, W) and (1, H, W) shapes.
    """

    def __init__(self, sigma_range=(0.5, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, lr, hr, mask):
        if random.random() > 0.5:
            sig = random.uniform(*self.sigma_range)
            if lr.ndim == 3:
                lr = gaussian_filter(lr, sigma=(0, sig, sig))
            else:
                lr = gaussian_filter(lr, sigma=sig)
        return lr, hr, mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, mask):
        for t in self.transforms:
            lr, hr, mask = t(lr, hr, mask)
        return lr, hr, mask

# ------------- Dataset -----------------------------
class SRDataset(Dataset):
    def __init__(self, metadata: pd.DataFrame, mean: float = None, std: float = None, transforms = None):
        self.samples    = metadata.reset_index(drop=True)
        self.mean       = mean
        self.std        = std
        self.transform  = transforms

    def __len__(self): return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        with rasterio.open(row["lr_path"]) as src:
            nodata_lr = src.nodata
            lr        = src.read(1).astype(np.float32)

        with rasterio.open(row["hr_path"]) as src:
            nodata_hr = src.nodata
            hr        = src.read(1).astype(np.float32)

        # Build mask BEFORE filling nodata
        hr_mask = (hr != nodata_hr).astype(np.float32) if nodata_hr is not None else (~np.isnan(hr)).astype(np.float32)

        # EDIT: Filling LR with mean instead of 0.0 for better radiometric consistency
        fill = float(self.mean) if self.mean is not None else 0.0

        hr = np.nan_to_num(hr, nan=fill) if nodata_hr is None else np.where(hr == nodata_hr, fill, hr)
        lr = np.nan_to_num(lr, nan=fill) if nodata_lr is None else np.where(lr == nodata_lr, fill, lr)

        # Apply augmentations (Applied to raw Celsius values)
        if self.transform is not None:
            lr, hr, hr_mask = self.transform(lr, hr, hr_mask)

        # Ensure contiguous for PyTorch
        lr, hr, hr_mask = np.ascontiguousarray(lr), np.ascontiguousarray(hr), np.ascontiguousarray(hr_mask)

        lr_t   = torch.from_numpy(lr).unsqueeze(0).float()
        hr_t   = torch.from_numpy(hr).unsqueeze(0).float()
        mask_t = torch.from_numpy(hr_mask).unsqueeze(0).float()

        if self.mean is not None and self.std is not None:
            lr_t = (lr_t - self.mean) / self.std
            hr_t = (hr_t - self.mean) / self.std

        return lr_t, hr_t, mask_t