"""
dataset.py — TIR Super-Resolution dataset
- LR/HR normalized independently
- AUX returned separately (FiLM input only)

Fixes applied:
  Bug 1 — aux is now cropped in train mode to match the hr patch window
  Bug 2 — zero-aux fallback is sized to HR spatial dims, not LR
"""

import random
import numpy as np
import pandas as pd
import rasterio
import torch
from torch.utils.data import Dataset
from scipy.ndimage import gaussian_filter

# ---------------- LULC ----------------
LULC_CLASSES = [10, 20, 30, 40, 50, 60, 70, 80, 90, 95, 100]
NUM_LULC_CLASSES = len(LULC_CLASSES)

LULC_LUT = {v: i for i, v in enumerate(LULC_CLASSES)}


def encode_lulc(lulc):
    encoded = np.full(lulc.shape, -1, dtype=np.int32)
    for raw, idx in LULC_LUT.items():
        encoded[lulc == raw] = idx
    return encoded


def one_hot_lulc(lulc_idx):
    h, w = lulc_idx.shape
    out = np.zeros((NUM_LULC_CLASSES, h, w), dtype=np.float32)
    for c in range(NUM_LULC_CLASSES):
        out[c] = (lulc_idx == c)
    return out


# ---------------- STATS ----------------
def compute_mean_std(metadata):
    s = ss = n = 0.0

    for _, row in metadata.iterrows():
        with rasterio.open(row["lr_path"]) as src:
            x = src.read(1).astype(np.float32)
            nodata = src.nodata

        m = np.isfinite(x)
        if nodata is not None:
            m &= ~np.isclose(x, nodata, atol=1e30)

        v = x[m]
        if len(v) == 0:
            continue

        s += v.sum()
        ss += (v ** 2).sum()
        n += len(v)

    mean = s / n
    std = np.sqrt(ss / n - mean**2)
    return float(mean), float(std)


def compute_aux_mean_std(metadata):
    sum_, sq_, count_ = None, None, None

    for _, row in metadata.iterrows():
        if "aux_path" not in row or pd.isna(row["aux_path"]):
            continue

        with rasterio.open(row["aux_path"]) as src:
            aux = src.read().astype(np.float32)

        cont = aux[:-1]

        if sum_ is None:
            sum_ = np.zeros(cont.shape[0])
            sq_ = np.zeros(cont.shape[0])
            count_ = np.zeros(cont.shape[0])

        for c in range(cont.shape[0]):
            v = cont[c]
            m = np.isfinite(v)
            vals = v[m]

            sum_[c] += vals.sum()
            sq_[c] += (vals ** 2).sum()
            count_[c] += len(vals)

    mean = sum_ / count_
    std = np.sqrt(np.maximum(sq_ / count_ - mean**2, 1e-6))
    return mean.astype(np.float32), std.astype(np.float32)


def compute_data_range(metadata, low_percentile=1.0, high_percentile=99.0, min_range=1e-6):
    """Compute data range from percentiles of valid HR pixels across the dataset."""
    values = []
    for _, row in metadata.iterrows():
        with rasterio.open(row["hr_path"]) as src:
            hr = src.read(1).astype(np.float32)
            nodata = src.nodata

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


# ---------------- DATASET ----------------
class SRDataset(Dataset):

    def __init__(
        self,
        metadata,
        mean=None,
        std=None,
        aux_mean=None,
        aux_std=None,
        patch_size=48,
        scale=4,
        transforms=None,
        is_train=True,
        use_aux=False,
    ):
        self.samples = metadata.reset_index(drop=True)

        self.mean = mean
        self.std = std

        self.aux_mean = aux_mean
        self.aux_std = aux_std

        self.patch_size = patch_size
        self.scale = scale
        self.transform = transforms
        self.is_train = is_train
        self.use_aux = use_aux

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

        aux = None
        if self.use_aux and "aux_path" in row and pd.notna(row["aux_path"]):
            with rasterio.open(row["aux_path"]) as src:
                aux = src.read().astype(np.float32)  # (C, H_hr, W_hr)

        fill = float(self.mean) if self.mean is not None else 0.0

        def clean(x, nd):
            m = np.isfinite(x)
            if nd is not None:
                m &= ~np.isclose(x, nd, atol=1e-3)
            return np.where(m, x, fill), m.astype(np.float32)

        lr, lm = clean(lr, nodata_lr)
        hr, hm = clean(hr, nodata_hr)

        # ---------------- cropping (train only) ----------------
        if self.is_train:
            hr_h, hr_w = hr.shape
            i = random.randint(0, hr_h - self.patch_size * self.scale)
            j = random.randint(0, hr_w - self.patch_size * self.scale)

            hr   = hr  [i : i + self.patch_size * self.scale,
                        j : j + self.patch_size * self.scale]
            lr   = lr  [i // self.scale : i // self.scale + self.patch_size,
                        j // self.scale : j // self.scale + self.patch_size]
            mask = hm  [i : i + self.patch_size * self.scale,
                        j : j + self.patch_size * self.scale]

            # FIX 1 — crop aux to the same HR patch window
            if aux is not None:
                aux = aux[
                    :,
                    i : i + self.patch_size * self.scale,
                    j : j + self.patch_size * self.scale,
                ]
        else:
            mask = hm  # validation/test: use full clean mask

        # ---------------- augmentations ----------------
        if self.transform:
            lr, hr, mask, aux = self.transform(lr, hr, mask, aux)

        # ---------------- convert to tensors ----------------
        lr   = torch.from_numpy(lr).unsqueeze(0).float()    # (1, H_lr, W_lr)
        hr   = torch.from_numpy(hr).unsqueeze(0).float()    # (1, H_hr, W_hr)
        mask = torch.from_numpy(mask).unsqueeze(0).float()  # (1, H_hr, W_hr)

        # ---------------- normalization ----------------
        if self.mean is not None:
            lr = (lr - self.mean) / self.std
            hr = (hr - self.mean) / self.std

        # ---------------- AUX handling ----------------
        if aux is not None:
            cont = aux[:-1]                                  # continuous bands
            if self.aux_mean is not None:
                cont = (cont - self.aux_mean[:, None, None]) / self.aux_std[:, None, None]

            lulc = encode_lulc(aux[-1].astype(np.int32))
            lulc = np.clip(lulc, 0, NUM_LULC_CLASSES - 1)

            aux_tensor = torch.cat([
                torch.from_numpy(cont).float(),
                torch.from_numpy(one_hot_lulc(lulc)).float(),
            ], dim=0)   # (C_cont + NUM_LULC_CLASSES, H_hr, W_hr)

        else:
            # FIX 2 — zeros must match HR spatial size, not LR
            aux_channels = (
                len(self.aux_mean) + NUM_LULC_CLASSES
                if self.aux_mean is not None
                else NUM_LULC_CLASSES
            )
            aux_tensor = torch.zeros(
                (aux_channels, hr.shape[-2], hr.shape[-1]),
                dtype=torch.float32,
            )

        return lr, hr, mask, aux_tensor