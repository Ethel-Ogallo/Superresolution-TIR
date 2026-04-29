"""
dataset.py — TIR Super-Resolution dataset creation and data utilities.
- Computes dataset statistics (mean, std, data range) for normalization and metrics.
- Defines data augmentations: geometric (flip/rotate), noise, blur.
- Implements SRDataset for loading LR/HR pairs, masks, and optional AUX data.
"""

import random
import numpy as np
import pandas as pd
import rasterio
import torch
import torch.nn.functional as F
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# ----------- AUXILIARY ENCODING FOR AUXILIARY ENCODER -----------
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
LULC_LUT = {v: i for i, v in enumerate(LULC_CLASSES)}
NUM_LULC_CLASSES = len(LULC_CLASSES)

def encode_lulc(lulc):
    """Convert ESA raw codes → contiguous indices."""
    encoded = np.full(lulc.shape, -1, dtype=np.int32)

    for raw, idx in LULC_LUT.items():
        encoded[lulc == raw] = idx

    return encoded

def one_hot_lulc(lulc_idx):
    """Convert index map → one-hot tensor (C,H,W)."""
    h, w = lulc_idx.shape
    out = np.zeros((NUM_LULC_CLASSES, h, w), dtype=np.float32)

    for c in range(NUM_LULC_CLASSES):
        out[c] = (lulc_idx == c)

    return out

# -------------- Data statistics ---------------
def compute_mean_std(metadata: pd.DataFrame):
    """Compute global mean and std from valid pixels across all LR images."""
    pixel_sum = pixel_sq = pixel_count = 0.0
    for _, row in metadata.iterrows():
        with rasterio.open(row["lr_path"]) as src:
            nodata = src.nodata
            lr     = src.read(1).astype(np.float32)

        valid_mask = ~np.isnan(lr)
        if nodata is not None:
            valid_mask &= ~np.isclose(lr, nodata, rtol=0, atol=1e30)
        valid = lr[valid_mask]
        if len(valid) == 0: continue

        pixel_sum   += valid.sum()
        pixel_sq    += (valid ** 2).sum()
        pixel_count += len(valid)

    mean = pixel_sum / pixel_count
    std  = np.sqrt(max(pixel_sq / pixel_count - mean ** 2, 0.0))
    print(f"  Train mean : {mean:.4f} °C  |  std : {std:.4f} °C")
    return float(mean), float(std)

def compute_data_range(metadata, low_percentile=1.0, high_percentile=99.0, min_range=1e-6):
    """Compute data range from percentiles of valid HR pixels across the dataset."""
    values = []
    for _, row in metadata.iterrows():
        with rasterio.open(row["hr_path"]) as src:
            hr = src.read(1).astype(np.float32)
            nodata = src.nodata
            
            # handle both nan and nodata
            valid = ~np.isnan(hr)
            if nodata is not None:
                valid &= ~np.isclose(hr, nodata, rtol=0, atol=1e30)
            
            hr_valid = hr[valid]
            if hr_valid.size > 0:
                values.append(hr_valid)

    if not values:
        raise ValueError("No valid HR pixels found to compute data range.")

    values = np.concatenate(values)
    low  = np.percentile(values, low_percentile)
    high = np.percentile(values, high_percentile)
    data_range = max(float(high - low), float(min_range))

    print(f"[INFO] Train data range ({low_percentile:.0f}-{high_percentile:.0f}th percentile): "
          f"{low:.2f}°C to {high:.2f}°C → range={data_range:.2f}°C")
    return data_range

# ------------ Data Augmentations ----------------
class GeoAugment:
    """Random horizontal/vertical flips and 90° rotations."""
    def __call__(self, lr, hr, mask, aux=None):
        if random.random() > 0.5:
            lr   = np.fliplr(lr).copy()
            hr   = np.fliplr(hr).copy()
            mask = np.fliplr(mask).copy()
            if aux is not None:
                aux = np.flip(aux, axis=2).copy()
        if random.random() > 0.5:
            lr   = np.flipud(lr).copy()
            hr   = np.flipud(hr).copy()
            mask = np.flipud(mask).copy()
            if aux is not None:
                aux = np.flip(aux, axis=1).copy()
        k = random.randint(0, 3)
        if k > 0:
            lr   = np.rot90(lr, k).copy()
            hr   = np.rot90(hr, k).copy()
            mask = np.rot90(mask, k).copy()
            if aux is not None:
                aux = np.rot90(aux, k, axes=(1, 2)).copy()
        return lr, hr, mask, aux


class TIRNoise:
    """Additive Gaussian noise with random scaling factor."""
    def __init__(self, std, p=0.5):
        self.std = std
        self.p   = p

    def __call__(self, lr, hr, mask, aux=None):
        if random.random() < self.p:
            multiplier = random.uniform(0.5, 1.5)
            noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
            lr = lr + noise
        return lr, hr, mask, aux


class BlurAugment:
    """Gaussian blur with random sigma applied to LR only."""
    def __init__(self, sigma_range=(0.5, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, lr, hr, mask, aux=None):
        if random.random() > 0.5:
            sig = random.uniform(*self.sigma_range)
            if lr.ndim == 3:
                lr = gaussian_filter(lr, sigma=(0, sig, sig))
            else:
                lr = gaussian_filter(lr, sigma=sig)
        return lr, hr, mask, aux

class Compose:
    """Compose multiple augmentations sequentially."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, mask, aux=None):
        for t in self.transforms:
            lr, hr, mask, aux = t(lr, hr, mask, aux)
        return lr, hr, mask, aux

# ------------- Dataset -----------------------------
class SRDataset(Dataset):
    """PyTorch Dataset for TIR Super-Resolution.
    Loads LR/HR image pairs, HR masks, and optional AUX data.
    Applies augmentations and returns tensors ready for model input.
    """
    def __init__(self, metadata, mean=None, std=None,patch_size=48, scale=4, 
                 transforms=None,is_train=True, use_aux=False):
        self.samples  = metadata.reset_index(drop=True)
        self.mean     = mean
        self.std      = std
        self.transform = transforms
        self.patch_size = patch_size
        self.scale    = scale
        self.is_train = is_train
        self.use_aux  = use_aux
    
    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        row = self.samples.iloc[idx]

        with rasterio.open(row["lr_path"]) as src:
            lr        = src.read(1).astype(np.float32)
            nodata_lr = src.nodata
        with rasterio.open(row["hr_path"]) as src:
            hr        = src.read(1).astype(np.float32)
            nodata_hr = src.nodata

        # load AUX data if available and enabled
        aux = None
        if self.use_aux and "aux_path" in row and pd.notna(row["aux_path"]):
            with rasterio.open(row["aux_path"]) as src:
                aux = src.read().astype(np.float32)  # (9, H, W)

        # Handle Nodata/NaN
        fill = float(self.mean) if self.mean is not None else 0.0

        def get_clean_data(arr, nd):
            mask = np.ones_like(arr, dtype=bool)
            if nd is not None:
                mask &= ~np.isclose(arr, nd, atol=1e-3)
            mask &= np.isfinite(arr)
            cleaned_arr = np.where(mask, arr, fill)
            return cleaned_arr, mask.astype(np.float32)

        lr, lr_mask = get_clean_data(lr, nodata_lr)
        hr, hr_mask = get_clean_data(hr, nodata_hr)

        lr = np.where(lr_mask == 1.0, lr, fill)
        hr = np.where(hr_mask == 1.0, hr, fill)

        if self.is_train:
            ih, iw = lr.shape[:2]
            for _ in range(20):
                iy   = random.randint(0, ih - self.patch_size)
                ix   = random.randint(0, iw - self.patch_size)
                iy_h = iy * self.scale
                ix_h = ix * self.scale
                ph_h = self.patch_size * self.scale
                target_mask = hr_mask[iy_h:iy_h+ph_h, ix_h:ix_h+ph_h]
                if target_mask.mean() > 0.2:
                    break

            lr      = lr[iy:iy+self.patch_size, ix:ix+self.patch_size]
            hr      = hr[iy_h:iy_h+ph_h, ix_h:ix_h+ph_h]
            hr_mask = target_mask

            # crop AUX to same spatial extent as HR if available
            if aux is not None:
                aux = aux[:, iy_h:iy_h+ph_h, ix_h:ix_h+ph_h]

        # Apply augmentations
        if self.transform is not None:
            # pass aux through transforms 
            lr, hr, hr_mask, aux = self.transform(lr, hr, hr_mask, aux)
            
        # to tensors
        lr      = np.ascontiguousarray(lr)
        hr      = np.ascontiguousarray(hr)
        hr_mask = np.ascontiguousarray(hr_mask)

        lr_t   = torch.from_numpy(lr).unsqueeze(0)
        hr_t   = torch.from_numpy(hr).unsqueeze(0)
        mask_t = torch.from_numpy(hr_mask).unsqueeze(0)

        if self.mean is not None and self.std is not None:
            lr_t = (lr_t - self.mean) / self.std
            hr_t = (hr_t - self.mean) / self.std

        # AUX PROCESSING 
        if aux is not None:
            aux = np.ascontiguousarray(aux)

            # split continuous + LULC
            cont_aux = aux[:-1]   # NDVI, NDWI, NDMI, bands...
            lulc_raw = aux[-1]     # ESA WorldCover MAP

            # encode LULC
            lulc_idx = encode_lulc(lulc_raw)
            lulc_onehot = one_hot_lulc(lulc_idx)

            # continuous tensor
            cont_t = torch.from_numpy(cont_aux)
            aux_min = cont_t.view(cont_t.shape[0], -1).min(1)[0][:, None, None]
            aux_max = cont_t.view(cont_t.shape[0], -1).max(1)[0][:, None, None]
            cont_t = (cont_t - aux_min) / (aux_max - aux_min + 1e-6)

            # LULC tensor
            lulc_t = torch.from_numpy(lulc_onehot)

            aux_t = torch.cat([cont_t, lulc_t], dim=0)

        else:
            # no AUX case
            aux_t = torch.zeros(
                (NUM_LULC_CLASSES, hr_t.shape[-2], hr_t.shape[-1])
            )

        return lr_t, hr_t, mask_t, aux_t  
                
