"""
Dataset for FastDiffSR TIR super-resolution
with auxiliary input conditioning and multiscale SPADE guidance.
"""

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
        repeat_channels=False,
        use_aux=False,
        aux_dir=None,
        transform=None,
    ):
        self.split = split
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.use_water_mask = use_water_mask
        self.repeat_channels = repeat_channels
        self.use_aux = use_aux

        # Auxiliary directories
        if aux_dir is not None:
            base = Path(aux_dir)
            self.aux_lr_dir = base
            self.aux_mid_dir = base.parent / "AUX_SPADE_MID"
            self.aux_hr_dir = base.parent / "AUX_SPADE_HR"
        else:
            self.aux_lr_dir = None
            self.aux_mid_dir = None
            self.aux_hr_dir = None

        # Normalisation statistics
        with open(stats_path) as f:
            stats = json.load(f)

        self.hr_mean = stats["hr"]["mean"]
        self.hr_std = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std = stats["lr"]["std"]

        # Temporal metadata
        metadata_path = self.patches_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # Patch directories
        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"
        self.wm_dir = self.patches_dir / split / "WM"

        self.files = sorted(
            f.name for f in self.hr_dir.glob("*.npy")
        )

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        # LR / HR
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)

        hr_mask = np.isfinite(hr).astype(np.float32)
        lr_mask = np.isfinite(lr).astype(np.float32)

        hr = np.where(hr_mask, hr, self.hr_mean)
        lr = np.where(lr_mask, lr, self.lr_mean)

        hr = (hr - self.hr_mean) / self.hr_std
        lr = (lr - self.lr_mean) / self.lr_std

        # auxiliary
        aux_lr = None
        aux_mid = None
        aux_hr = None

        if self.use_aux:
            if self.aux_lr_dir is None:
                raise ValueError("use_aux=True but aux_dir was not provided.")
            aux_lr = np.load(self.aux_lr_dir / fname).astype(np.float32)
            aux_mid = np.load(self.aux_mid_dir / fname ).astype(np.float32)
            aux_hr = np.load(self.aux_hr_dir / fname).astype(np.float32)

        # water mask
        wm = np.load(self.wm_dir / fname).astype(np.float32)

        # Apply synchronized spatial augmentations
        if self.transform:
            lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, wm = (self.transform(lr, hr, hr_mask,
                                                                           aux_lr, aux_mid, aux_hr,wm,))

        # Validate auxiliary dimensions
        if self.use_aux:
            assert aux_lr.shape[-2:] == (64, 64), aux_lr.shape
            assert aux_mid.shape[-2:] == (128, 128), aux_mid.shape
            assert aux_hr.shape[-2:] == (256, 256), aux_hr.shape

        # Build sample
        if self.repeat_channels:
            hr = np.stack([hr] * 3, axis=0)
            lr = np.stack([lr] * 3, axis=0)
        else:
            hr = hr[None]
            lr = lr[None]

        sample = {
            "lr": torch.from_numpy(lr).float()[None],
            "hr": torch.from_numpy(hr).float()[None],
            "hr_mask": torch.from_numpy(hr_mask).float()[None],
            "water_mask": torch.from_numpy(wm).float()[None],
            "fname": fname,
        }

        if self.use_aux:
            sample["aux_lr"] = torch.from_numpy(aux_lr).float()
            sample["aux_mid"] = torch.from_numpy(aux_mid).float()
            sample["aux_hr"] = torch.from_numpy(aux_hr).float()

        # 7. Temporal metadata
        key = Path(fname).stem
        meta = self.metadata.get(key, {})

        time_gap_hours = float(meta.get("time_gap_hours", 0.0))
        date_gap_days = float(meta.get("date_gap_days", 0.0))

        sample["time_gap_hours"] = torch.tensor(time_gap_hours, dtype=torch.float32)
        sample["date_gap_days"] = torch.tensor(date_gap_days, dtype=torch.float32)

        # 8. Temporal channels at HR resolution
        H, W = hr.shape[-2:]

        time_channels = np.stack([
            np.full((H, W), time_gap_hours / 24.0, dtype=np.float32),
            np.full((H, W), date_gap_days / 365.0, dtype=np.float32),
        ], axis=0)

        sample["time_channels"] = torch.from_numpy(time_channels).float()
        return sample