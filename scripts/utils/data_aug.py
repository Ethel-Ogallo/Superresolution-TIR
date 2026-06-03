"""
augmentations.py — Spatial augmentations for TIR Super-Resolution 

"""

import random
import numpy as np
from scipy.ndimage import gaussian_filter


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        if random.random() < self.p:
            # Axis -1 is always Width for [H, W], [1, H, W], or [C, H, W]
            lr = np.flip(lr, axis=-1).copy()
            hr = np.flip(hr, axis=-1).copy()
            hr_mask = np.flip(hr_mask, axis=-1).copy()

            if water_mask is not None:
                water_mask = np.flip(water_mask, axis=-1).copy()
            if aux_lr is not None:
                aux_lr = np.flip(aux_lr, axis=-1).copy()
            if aux_mid is not None:
                aux_mid = np.flip(aux_mid, axis=-1).copy()
            if aux_hr is not None:
                aux_hr = np.flip(aux_hr, axis=-1).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        if random.random() < self.p:
            # Axis -2 is always Height for [H, W], [1, H, W], or [C, H, W]
            lr = np.flip(lr, axis=-2).copy()
            hr = np.flip(hr, axis=-2).copy()
            hr_mask = np.flip(hr_mask, axis=-2).copy()

            if water_mask is not None:
                water_mask = np.flip(water_mask, axis=-2).copy()
            if aux_lr is not None:
                aux_lr = np.flip(aux_lr, axis=-2).copy()
            if aux_mid is not None:
                aux_mid = np.flip(aux_mid, axis=-2).copy()
            if aux_hr is not None:
                aux_hr = np.flip(aux_hr, axis=-2).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


class RandomRotation90:
    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        k = random.randint(0, 3)
        if k == 0:
            return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask

        # Explicitly rotate on the last two spatial axes (-2, -1) to support multi-channel arrays cleanly
        lr = np.rot90(lr, k, axes=(-2, -1)).copy()
        hr = np.rot90(hr, k, axes=(-2, -1)).copy()
        hr_mask = np.rot90(hr_mask, k, axes=(-2, -1)).copy()

        if water_mask is not None:
            water_mask = np.rot90(water_mask, k, axes=(-2, -1)).copy()
        if aux_lr is not None:
            aux_lr = np.rot90(aux_lr, k, axes=(-2, -1)).copy()
        if aux_mid is not None:
            aux_mid = np.rot90(aux_mid, k, axes=(-2, -1)).copy()
        if aux_hr is not None:
            aux_hr = np.rot90(aux_hr, k, axes=(-2, -1)).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


class TIRNoise:
    def __init__(self, std, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        if random.random() < self.p:
            multiplier = random.uniform(0.5, 1.5)
            noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
            lr = lr + noise

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


class BlurAugment:
    def __init__(self, sigma_range=(0.5, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        if random.random() > 0.5:
            sig = random.uniform(*self.sigma_range)

            if lr.ndim == 3:
                # Apply blur only along spatial dimensions, ignoring channel dimension (index 0)
                lr = gaussian_filter(lr, sigma=(0, sig, sig))
            else:
                lr = gaussian_filter(lr, sigma=sig)

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, hr_mask, aux_lr=None, aux_mid=None, aux_hr=None, water_mask=None):

        for t in self.transforms:
            lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask = t(
                lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask
            )

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, water_mask


def train_transforms():
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        TIRNoise(std=0.01, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.5)),
    ])