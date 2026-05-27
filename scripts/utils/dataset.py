import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


# ============================
# SR DATASET (FIXED FOR SPADE DIRECT)
# ============================

class SRDataset(Dataset):

    def __init__(
        self,
        split,
        patches_dir,
        stats_path,
        use_water_mask=False,
        use_aux=True,
        aux_dir=None,
        transform=None,
        repeat_channels=False,
    ):

        self.split = split
        self.patches_dir = Path(patches_dir)
        self.transform = transform
        self.repeat_channels = repeat_channels

        self.use_water_mask = use_water_mask
        self.use_aux = use_aux

        # -----------------------
        # AUX PATHS (CORRECTED)
        # -----------------------
        if aux_dir is not None:
            base = Path(aux_dir)
            self.aux_lr_dir   = base
            self.aux_mid_dir  = base.parent / "AUX_SPADE_MID"
            self.aux_hr_dir   = base.parent / "AUX_SPADE_HR"

        # -----------------------
        # STATS
        # -----------------------
        with open(stats_path) as f:
            stats = json.load(f)

        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        # -----------------------
        # MAIN PATHS
        # -----------------------
        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"
        self.wm_dir = self.patches_dir / split / "WM"

        self.files = sorted([f.name for f in self.hr_dir.glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):

        fname = self.files[idx]

        # -----------------------
        # LOAD LR / HR
        # -----------------------
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)

        hr_mask = np.isfinite(hr).astype(np.float32)
        lr_mask = np.isfinite(lr).astype(np.float32)

        hr = np.where(hr_mask, hr, self.hr_mean)
        lr = np.where(lr_mask, lr, self.lr_mean)

        # normalize
        hr = (hr - self.hr_mean) / self.hr_std
        lr = (lr - self.lr_mean) / self.lr_std

        # -----------------------
        # AUX (DIRECT + SPADE)
        # -----------------------
        aux_lr = None
        aux_mid = None
        aux_hr = None

        if self.use_aux:
            aux_lr = np.load(self.aux_lr_dir / fname).astype(np.float32)
            aux_mid = np.load(self.aux_mid_dir / fname).astype(np.float32)
            aux_hr = np.load(self.aux_hr_dir / fname).astype(np.float32)

        # -----------------------
        # TRANSFORM
        # -----------------------
        if self.transform:
            lr, hr, hr_mask, aux_lr, aux_mid, aux_hr = self.transform(
                lr, hr, hr_mask, aux_lr, aux_mid, aux_hr
            )

        # -----------------------
        # TO TENSOR
        # -----------------------
        lr = torch.from_numpy(lr).float()[None]
        hr = torch.from_numpy(hr).float()[None]

        hr_mask = torch.from_numpy(hr_mask).float()[None]

        sample = {
            "lr": lr,
            "hr": hr,
            "hr_mask": hr_mask,
            "fname": fname,
        }

        # -----------------------
        # AUX OUTPUTS (CRITICAL)
        # -----------------------
        if aux_lr is not None:
            sample["aux_lr"] = torch.from_numpy(aux_lr).float()

        if aux_mid is not None:
            sample["aux_mid"] = torch.from_numpy(aux_mid).float()

        if aux_hr is not None:
            sample["aux_hr"] = torch.from_numpy(aux_hr).float()

        # -----------------------
        # WATER MASK
        # -----------------------
        if self.use_water_mask:
            wm = np.load(self.wm_dir / fname).astype(np.float32)
            sample["water_mask"] = torch.from_numpy(wm).float()[None]

        return sample