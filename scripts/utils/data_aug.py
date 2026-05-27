"""
augmentations.py — Spatial augmentations for TIR Super-Resolution 

Rules:
- All transforms applied identically to LR, HR and hr_mask to preserve spatial correspondence
- LR transforms use scale-correct versions of spatial ops
"""

import random
import numpy as np
from scipy.ndimage import gaussian_filter


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None):

        if random.random() < self.p:
            lr = np.fliplr(lr).copy()
            hr = np.fliplr(hr).copy()
            hr_mask = np.fliplr(hr_mask).copy()

            if aux_lr is not None:
                aux_lr = np.fliplr(aux_lr).copy()
            if aux_mid is not None:
                aux_mid = np.fliplr(aux_mid).copy()
            if aux_hr is not None:
                aux_hr = np.fliplr(aux_hr).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None):

        if random.random() < self.p:
            lr = np.flipud(lr).copy()
            hr = np.flipud(hr).copy()
            hr_mask = np.flipud(hr_mask).copy()

            if aux_lr is not None:
                aux_lr = np.flipud(aux_lr).copy()
            if aux_mid is not None:
                aux_mid = np.flipud(aux_mid).copy()
            if aux_hr is not None:
                aux_hr = np.flipud(aux_hr).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


class RandomRotation90:
    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None):

        k = random.randint(0, 3)
        if k == 0:
            return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr

        lr = np.rot90(lr, k).copy()
        hr = np.rot90(hr, k).copy()
        hr_mask = np.rot90(hr_mask, k).copy()

        if aux_lr is not None:
            aux_lr = np.rot90(aux_lr, k, axes=(1, 2)).copy()
        if aux_mid is not None:
            aux_mid = np.rot90(aux_mid, k, axes=(1, 2)).copy()
        if aux_hr is not None:
            aux_hr = np.rot90(aux_hr, k, axes=(1, 2)).copy()

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


class TIRNoise:
    def __init__(self, std, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None):

        if random.random() < self.p:
            multiplier = random.uniform(0.5, 1.5)
            noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
            lr = lr + noise

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


class BlurAugment:
    def __init__(self, sigma_range=(0.5, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, lr, hr, hr_mask,
                 aux_lr=None, aux_mid=None, aux_hr=None):

        if random.random() > 0.5:
            sig = random.uniform(*self.sigma_range)

            if lr.ndim == 3:
                lr = gaussian_filter(lr, sigma=(0, sig, sig))
            else:
                lr = gaussian_filter(lr, sigma=sig)

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, hr_mask, aux_lr=None, aux_mid=None, aux_hr=None):

        for t in self.transforms:
            lr, hr, hr_mask, aux_lr, aux_mid, aux_hr = t(
                lr, hr, hr_mask, aux_lr, aux_mid, aux_hr
            )

        return lr, hr, hr_mask, aux_lr, aux_mid, aux_hr


def train_transforms():
    """
    Standard spatial augmentations 
    Safe for SISR — all transforms preserve HR/LR spatial correspondence.
    """
    return Compose([
        RandomHorizontalFlip(p=0.5),
        RandomVerticalFlip(p=0.5),
        RandomRotation90(),
        TIRNoise(std=0.01, p=0.5),
        BlurAugment(sigma_range=(0.5, 1.5)),
    ])


