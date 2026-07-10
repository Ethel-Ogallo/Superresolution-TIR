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
        # Input modes
        use_water_mask=False,
        repeat_channels=False,   # keep for baseline compatibility
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
        self.aux_dir = Path(aux_dir) if aux_dir is not None else None

        # STATS
        with open(stats_path) as f:
            stats = json.load(f)

        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        metadata_path = self.patches_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # PATHS
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

        aux = None
        if self.use_aux and self.aux_dir is not None:
            aux = np.load(self.aux_dir / fname).astype(np.float32)

        # PASS AUX INTO TRANSFORM
        if self.transform:
            lr, hr, hr_mask, aux = self.transform(lr, hr, hr_mask, aux)

        # channel handling
        if self.repeat_channels:
            hr = np.stack([hr] * 3, axis=0)
            lr = np.stack([lr] * 3, axis=0)
        else:
            hr = hr[None]
            lr = lr[None]

        hr       = torch.from_numpy(hr).float()
        lr       = torch.from_numpy(lr).float()
        hr_mask  = torch.from_numpy(hr_mask)[None].float()
        lr_mask  = torch.from_numpy(lr_mask)[None].float()

        water_mask = None
        if self.use_water_mask:
            wm = np.load(self.wm_dir / fname).astype(np.float32)
            water_mask = torch.from_numpy(wm)[None].float()

        sample = {
            "lr":      lr,
            "hr":      hr,
            "hr_mask": hr_mask,
            "lr_mask": lr_mask,
            "fname":   fname,
        }

        if water_mask is not None:
            sample["water_mask"] = water_mask

        if aux is not None:
            sample["aux"] = torch.from_numpy(aux).float()
        
        # time channels
        key  = Path(fname).stem
        meta = self.metadata.get(key, {})

        # sample["lr_time"] = torch.tensor(float(meta.get("lr_time", 10.0)), dtype=torch.float32,)
        sample["time_gap_hours"] = torch.tensor(float(meta.get("time_gap_hours", 0.0)), dtype=torch.float32,)
        sample["date_gap_days"] = torch.tensor(float(meta.get("date_gap_days", 0.0)), dtype=torch.float32,)

        # ── build time channels for model input ─────────────────────
        H, W = lr.shape[-2], lr.shape[-1]

        # lr_time = float(meta.get("lr_time", 10.0))
        time_gap_hours = float(meta.get("time_gap_hours", 0.0))
        date_gap_days = float(meta.get("date_gap_days", 0.0))

        time_channels = np.stack([
            # np.full((H, W), lr_time / 24.0, dtype=np.float32),
            np.full((H, W), time_gap_hours / 24.0, dtype=np.float32),
            np.full((H, W), date_gap_days / 365.0, dtype=np.float32),
        ], axis=0)

        sample["time_channels"] = torch.from_numpy(time_channels).float()

        return sample