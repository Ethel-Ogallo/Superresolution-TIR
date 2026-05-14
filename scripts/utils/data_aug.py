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

    def __call__(self, lr, hr, hr_mask, aux=None):
        if random.random() < self.p:
            lr      = np.fliplr(lr).copy()
            hr      = np.fliplr(hr).copy()
            hr_mask = np.fliplr(hr_mask).copy()
            if aux is not None:
                aux = aux[:, :, ::-1].copy()  # flip W axis for (C, H, W)
        return lr, hr, hr_mask, aux


class RandomVerticalFlip:
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, lr, hr, hr_mask, aux=None):
        if random.random() < self.p:
            lr      = np.flipud(lr).copy()
            hr      = np.flipud(hr).copy()
            hr_mask = np.flipud(hr_mask).copy()
            if aux is not None:
                aux = aux[:, ::-1, :].copy()  # flip H axis for (C, H, W)
        return lr, hr, hr_mask, aux


class RandomRotation90:
    def __call__(self, lr, hr, hr_mask, aux=None):
        k = random.randint(0, 3)
        if k == 0:
            return lr, hr, hr_mask, aux
        lr      = np.rot90(lr,      k).copy()
        hr      = np.rot90(hr,      k).copy()
        hr_mask = np.rot90(hr_mask, k).copy()
        if aux is not None:
            aux = np.rot90(aux, k, axes=(1, 2)).copy()  # rotate on H, W axes
        return lr, hr, hr_mask, aux


class TIRNoise:
    def __init__(self, std, p=0.5):
        self.std = std
        self.p = p

    def __call__(self, lr, hr, hr_mask, aux=None):
        if random.random() < self.p:
            multiplier = random.uniform(0.5, 1.5)
            noise = np.random.randn(*lr.shape).astype(np.float32) * (self.std * multiplier)
            lr = lr + noise
        return lr, hr, hr_mask, aux  # aux unchanged


class BlurAugment:
    def __init__(self, sigma_range=(0.5, 1.5)):
        self.sigma_range = sigma_range

    def __call__(self, lr, hr, hr_mask, aux=None):
        if random.random() > 0.5:
            sig = random.uniform(*self.sigma_range)
            if lr.ndim == 3:
                lr = gaussian_filter(lr, sigma=(0, sig, sig))
            else:
                lr = gaussian_filter(lr, sigma=sig)
        return lr, hr, hr_mask, aux  # aux unchanged


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, lr, hr, hr_mask, aux=None):
        for t in self.transforms:
            lr, hr, hr_mask, aux = t(lr, hr, hr_mask, aux)
        return lr, hr, hr_mask, aux


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


