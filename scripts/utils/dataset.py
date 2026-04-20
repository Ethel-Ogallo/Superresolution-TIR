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

def compute_data_range(
    metadata: pd.DataFrame,
    low_percentile: float = 1.0,
    high_percentile: float = 99.0,
    min_range: float = 1e-6,
) -> float:
    """Estimate a robust HR data range from train-only patches using percentiles.
    Uses 1st-99th percentile to avoid outlier skew from nodata edges or hot pixels.
    """
    values = []
    for _, row in metadata.iterrows():
        with rasterio.open(row["hr_path"]) as src:
            hr = src.read(1).astype(np.float32)
            nodata = src.nodata
            if nodata is not None:
                valid = hr[hr != nodata]
            else:
                valid = hr[~np.isnan(hr)]
            if valid.size > 0:
                values.append(valid)

    if not values:
        raise ValueError("No valid HR pixels found to compute data range.")

    values = np.concatenate(values)
    low  = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)
    data_range = max(float(high - low), float(min_range))

    print(
        f"[INFO] Train data range ({low_percentile:.0f}-{high_percentile:.0f}th percentile): "
        f"{low:.2f}°C to {high:.2f}°C → range={data_range:.2f}°C"
    )
    return data_range

# ------------ Data Augmentations ----------------
#TODO: utilize kornia or pytorch to do the augmentations

class GeoAugment:
    """Geometric transforms: flips + 90° rotations."""
#TODO: why not just use torchvision.transforms.RandomHorizontalFlip and RandomVerticalFlip and RandomRotation? 
# Because we need to apply the same transforms to both LR and HR (and mask), and torchvision's functional API is a bit clunky for that. 
# This custom class allows us to easily apply the same random transform to all three arrays in a consistent way.
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
    std is estimated from the LR (Landsat-8) patches.
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
    Optional inter-sensor bias applied to LR only. ??sensor bias??
    """
    def __init__(self, range_c=1.0):  #sensor_bias_range=0.5
        self.range_c = range_c
        # self.sensor_bias = sensor_bias_range

    def __call__(self, lr, hr, mask):
        shift = random.uniform(-self.range_c, self.range_c)
        # bias = random.uniform(-self.sensor_bias, self.sensor_bias)
        return lr + shift , hr + shift , mask  #+ bias


class ContrastScaling:
    """
    Scales thermal gradients around each image's own mean. ??physically meaningful??
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
    def __init__(self, sigma_range=(0.5, 1.5)): #??how to choose sigma range??
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
    def __init__(self, metadata, mean=None, std=None, 
                 patch_size=64, scale=4, transforms=None, 
                 is_train=True):
        self.samples = metadata.reset_index(drop=True)
        self.mean = mean
        self.std = std
        self.transform = transforms
        self.patch_size = patch_size
        self.scale = scale
        self.is_train = is_train
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        with rasterio.open(row["lr_path"]) as src:
            lr = src.read(1).astype(np.float32)
            nodata_lr = src.nodata
        with rasterio.open(row["hr_path"]) as src:
            hr = src.read(1).astype(np.float32)
            nodata_hr = src.nodata

        # Handle Nodata/NaN 
        fill = float(self.mean) if self.mean is not None else 0.0

        hr_mask = (hr != np.float32(nodata_hr)).astype(np.float32) if nodata_hr is not None else (~np.isnan(hr)).astype(np.float32)
        lr = np.nan_to_num(lr, nan=fill) if nodata_lr is None else np.where(lr == np.float32(nodata_lr), fill, lr)
        hr = np.nan_to_num(hr, nan=fill) if nodata_hr is None else np.where(hr == np.float32(nodata_hr), fill, hr)

        # Patch Extraction (Only if Training)
        if self.is_train:
            ih, iw = lr.shape[:2]
            # Ensure we don't pick a starting point that goes out of bounds
            iy = random.randint(0, ih - self.patch_size)
            ix = random.randint(0, iw - self.patch_size)
            
            lr = lr[iy : iy + self.patch_size, ix : ix + self.patch_size]
            
            # Scale coordinates for HR and Mask
            iy_h, ix_h = iy * self.scale, ix * self.scale
            ph_h = self.patch_size * self.scale
            hr = hr[iy_h : iy_h + ph_h, ix_h : ix_h + ph_h]
            hr_mask = hr_mask[iy_h : iy_h + ph_h, ix_h : ix_h + ph_h]

        # Apply Remaining Augmentations 
        if self.transform is not None:
            lr, hr, hr_mask = self.transform(lr, hr, hr_mask)

        # to tensors and normalize
        lr, hr, hr_mask = np.ascontiguousarray(lr), np.ascontiguousarray(hr), np.ascontiguousarray(hr_mask)
        
        lr_t = torch.from_numpy(lr).unsqueeze(0)
        hr_t = torch.from_numpy(hr).unsqueeze(0)
        mask_t = torch.from_numpy(hr_mask).unsqueeze(0)

        if self.mean is not None and self.std is not None:
            lr_t = (lr_t - self.mean) / self.std
            hr_t = (hr_t - self.mean) / self.std

        return lr_t, hr_t, mask_t

