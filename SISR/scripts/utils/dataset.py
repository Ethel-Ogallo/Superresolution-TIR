# scripts/utils/dataset.py

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SRDataset(Dataset):

    def __init__(
        self,
        split,
        patches_dir,
        stats_path,
        use_water_mask=False,
        repeat_channels=True,
        transform=None,
    ):
        self.split = split
        self.patches_dir = Path(patches_dir)
        self.use_water_mask = use_water_mask
        self.repeat_channels = repeat_channels
        self.transform = transform

        with open(stats_path) as f:
            stats = json.load(f)

        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"
        self.wm_dir = self.patches_dir / split / "WM"

        self.files = sorted([f.name for f in self.hr_dir.glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        hr = np.load(self.hr_dir / fname).astype(np.float32)
        lr = np.load(self.lr_dir / fname).astype(np.float32)

        hr_mask = np.isfinite(hr).astype(np.float32)
        lr_mask = np.isfinite(lr).astype(np.float32)

        hr = np.where(hr_mask, hr, self.hr_mean)
        lr = np.where(lr_mask, lr, self.lr_mean)

        hr = (hr - self.hr_mean) / self.hr_std
        lr = (lr - self.lr_mean) / self.lr_std

        if self.transform:
            lr, hr, hr_mask = self.transform(lr, hr, hr_mask)

        if self.repeat_channels:
            hr = np.stack([hr]*3, axis=0)
            lr = np.stack([lr]*3, axis=0)
        else:
            hr = hr[None]
            lr = lr[None]

        hr = torch.from_numpy(hr).float()
        lr = torch.from_numpy(lr).float()
        hr_mask = torch.from_numpy(hr_mask)[None].float()
        lr_mask = torch.from_numpy(lr_mask)[None].float()

        # water mask (optional)
        water_mask = None
        if self.use_water_mask:
            wm = np.load(self.wm_dir / fname).astype(np.float32)
            water_mask = torch.from_numpy(wm)[None].float()

        sample = {
            "lr": lr,
            "hr": hr,
            "hr_mask": hr_mask,
            "lr_mask": lr_mask,
            "fname": fname,
        }

        if water_mask is not None:
            sample["water_mask"] = water_mask

        return sample