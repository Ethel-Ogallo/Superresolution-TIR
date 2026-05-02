"""
augmentations.py — Spatial augmentations for TIR Super-Resolution (Phase 2)

Rules:
- All transforms applied identically to LR, HR and hr_mask
  to preserve spatial correspondence
- Only label-preserving transforms (no colour jitter, no independent crops)
- LR transforms use scale-correct versions of spatial ops
- Augmentations applied to train only — dataset handles this
"""

import random
import numpy as np
from scipy.ndimage import gaussian_filter


class RandomHorizontalFlip:
    """Flip LR, HR and mask horizontally with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask):
        if random.random() < self.p:
            lr      = np.fliplr(lr).copy()       # (H, W) or (C, H, W)
            hr      = np.fliplr(hr).copy()
            hr_mask = np.fliplr(hr_mask).copy()
        return lr, hr, hr_mask


class RandomVerticalFlip:
    """Flip LR, HR and mask vertically with probability p."""
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask):
        if random.random() < self.p:
            lr      = np.flipud(lr).copy()
            hr      = np.flipud(hr).copy()
            hr_mask = np.flipud(hr_mask).copy()
        return lr, hr, hr_mask


class RandomRotation90:
    """Random 90° rotation (0, 90, 180, 270) applied identically to LR, HR and mask."""
    def __call__(self, lr, hr, hr_mask):
        k = random.randint(0, 3)
        if k == 0:
            return lr, hr, hr_mask
        lr      = np.rot90(lr,      k).copy()
        hr      = np.rot90(hr,      k).copy()
        hr_mask = np.rot90(hr_mask, k).copy()
        return lr, hr, hr_mask

# class TIRNoise:
#     """Additive Gaussian noise with random scaling factor."""
#     def __init__(self, std, p=0.5):
#         self.std = std
#         self.p   = p

#     def __call__(self, lr, hr, mask, aux=None):
#         if random.random() < self.p:
#             multiplier = random.uniform(0.5, 1.5)
#             noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
#             lr = lr + noise
#         return lr, hr, mask, aux


# class BlurAugment:
#     """Gaussian blur with random sigma applied to LR only."""
#     def __init__(self, sigma_range=(0.5, 1.5)):
#         self.sigma_range = sigma_range

#     def __call__(self, lr, hr, mask, aux=None):
#         if random.random() > 0.5:
#             sig = random.uniform(*self.sigma_range)
#             if lr.ndim == 3:
#                 lr = gaussian_filter(lr, sigma=(0, sig, sig))
#             else:
#                 lr = gaussian_filter(lr, sigma=sig)
#         return lr, hr, mask, aux

# Random cropping for both HR and LR uisng basicsr's paired random crop
# from basicsr.data.util import paired_random_crop
# def random_crop(lr, hr, hr_mask, crop_size=128, scale=4):
#     """Randomly crop a HR patch and the corresponding LR patch."""
#     lr_crop, hr_crop, mask_crop = paired_random_crop(
#         lr, hr, hr_mask, crop_size, scale
#     )
#     return lr_crop, hr_crop, mask_crop

class Compose:
    """Apply a sequence of transforms."""
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, hr_mask):
        for t in self.transforms:
            lr, hr, hr_mask = t(lr, hr, hr_mask)
        return lr, hr, hr_mask


def train_transforms():
    """
    Standard spatial augmentations 
    Safe for SISR — all transforms preserve HR/LR spatial correspondence.
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
    ])


### Quick test
# if __name__ == "__main__":
#     import torch
#     from pathlib import Path
#     from dataset_copy import SRDataset

#     PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
#     STATS_PATH  = PATCHES_DIR / "stats.json"

#     transforms = train_transforms()

#     ds = SRDataset(
#         split="train",
#         patches_dir=PATCHES_DIR,
#         stats_path=STATS_PATH,
#         repeat_channels=True,
#         transform=transforms,
#     )

#     sample = ds[0]
#     lr, hr, hr_mask = sample["lr"], sample["hr"], sample["hr_mask"]

#     print("Augmentation test:")
#     print(f"  lr      : {tuple(lr.shape)}")
#     print(f"  hr      : {tuple(hr.shape)}")
#     print(f"  hr_mask : {tuple(hr_mask.shape)}")
#     print("  Shapes preserved after augmentation")


# how it plugs later in training loop:
# from augmentations import get_train_transforms
# from dataset import SRDataset

# train_ds = SRDataset(
#     split="train",
#     patches_dir=PATCHES_DIR,
#     stats_path=STATS_PATH,
#     repeat_channels=True,
#     transform=train_transforms(),  # only for train
# )

# val_ds = SRDataset(
#     split="val",
#     patches_dir=PATCHES_DIR,
#     stats_path=STATS_PATH,
#     repeat_channels=True,
#     transform=None,                    # no augmentation for val/test
# )