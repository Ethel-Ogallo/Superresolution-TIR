"""
dataset.py — SR Dataset for TIR Super-Resolution

Loads:
  - LR TIR patch          (normalized)
  - HR TIR patch          (normalized)
  - HR valid mask         (1 where HR is not NaN)
  - aux_lr                (20ch aux at LR resolution, direct input)
  - aux_mid               (20ch aux at 128px, SPADE mid conditioning)
  - aux_hr                (20ch aux at 256px, SPADE HR conditioning)
  - water_mask            (1 where pixel is water/river)
  - lr_time               (LR acquisition hour, decimal float e.g. 10.37)
  - time_gap_hours        (abs hour difference between LR and HR acquisition)
  - date_gap_days         (calendar day difference between LR and HR)

Time fields are used ONLY in the loss function (time-aware gradient loss).
They are not fed into the model.

Aux channel layout (20ch, fixed at patch creation):
  [0:5]  spectral bands 1-5   continuous [0, 1]
  [5]    NDVI                 continuous [-1, 1]
  [6]    NDWI                 continuous [-1, 1]
  [7]    NDMI                 continuous [-1, 1]
  [8:19] LULC one-hot (11cls) binary     {0, 1}
  [19]   DEM                  continuous [0, 1]
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
        use_aux=True,
        aux_dir=None,
        transform=None,
        repeat_channels=False,
    ):
        self.split           = split
        self.patches_dir     = Path(patches_dir)
        self.transform       = transform
        self.repeat_channels = repeat_channels
        self.use_water_mask  = use_water_mask
        self.use_aux         = use_aux

        # ── AUX PATHS ─────────────────────────────────────────
        # aux_dir points to the LR-resolution aux folder.
        # Mid and HR SPADE aux are siblings of that folder.
        if aux_dir is not None:
            base = Path(aux_dir)
            self.aux_lr_dir  = base
            self.aux_mid_dir = base.parent / "AUX_SPADE_MID"
            self.aux_hr_dir  = base.parent / "AUX_SPADE_HR"
        else:
            self.aux_lr_dir  = None
            self.aux_mid_dir = None
            self.aux_hr_dir  = None

        # ── NORMALIZATION STATS ───────────────────────────────
        with open(stats_path) as f:
            stats = json.load(f)

        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        # ── METADATA ─────────────────────────────────────────
        # Loaded once at init — used in __getitem__ for time fields.
        # metadata.json lives at the root of patches_dir.
        metadata_path = self.patches_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # ── MAIN PATCH PATHS ──────────────────────────────────
        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"
        self.wm_dir = self.patches_dir / split / "WM"

        self.files = sorted([f.name for f in self.hr_dir.glob("*.npy")])

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]

        # ── LR / HR ───────────────────────────────────────────
        lr = np.load(self.lr_dir / fname).astype(np.float32)
        hr = np.load(self.hr_dir / fname).astype(np.float32)

        # Valid pixel masks (1 where not NaN)
        hr_mask = np.isfinite(hr).astype(np.float32)
        lr_mask = np.isfinite(lr).astype(np.float32)

        # Fill NaN with mean before normalizing
        hr = np.where(hr_mask, hr, self.hr_mean)
        lr = np.where(lr_mask, lr, self.lr_mean)

        # Normalize to zero mean unit variance
        hr = (hr - self.hr_mean) / self.hr_std
        lr = (lr - self.lr_mean) / self.lr_std

        # ── AUX (DIRECT INPUT + SPADE CONDITIONING) ───────────
        aux_lr  = None
        aux_mid = None
        aux_hr  = None

        if self.use_aux and self.aux_lr_dir is not None:
            aux_lr  = np.load(self.aux_lr_dir  / fname).astype(np.float32)
            aux_mid = np.load(self.aux_mid_dir / fname).astype(np.float32)
            aux_hr  = np.load(self.aux_hr_dir  / fname).astype(np.float32)

        # ── AUGMENTATION ──────────────────────────────────────
        if self.transform:
            lr, hr, hr_mask, aux_lr, aux_mid, aux_hr = self.transform(
                lr, hr, hr_mask, aux_lr, aux_mid, aux_hr
            )

        # ── TO TENSOR ─────────────────────────────────────────
        # LR and HR are single-channel — add channel dim
        lr      = torch.from_numpy(lr).float()[None]       # [1, H, W]
        hr      = torch.from_numpy(hr).float()[None]       # [1, H, W]
        hr_mask = torch.from_numpy(hr_mask).float()[None]  # [1, H, W]

        # ── BASE SAMPLE ───────────────────────────────────────
        sample = {
            "lr":      lr,
            "hr":      hr,
            "hr_mask": hr_mask,
            "fname":   fname,
        }

        # ── AUX TENSORS ───────────────────────────────────────
        # aux_lr:  [20, 64,  64]  — concatenated with LR as direct input
        # aux_mid: [20, 128, 128] — conditions SPADE at 128px
        # aux_hr:  [20, 256, 256] — conditions SPADE at 256px
        if aux_lr is not None:
            sample["aux_lr"]  = torch.from_numpy(aux_lr).float()
            sample["aux_mid"] = torch.from_numpy(aux_mid).float()
            sample["aux_hr"]  = torch.from_numpy(aux_hr).float()

        # ── WATER MASK ────────────────────────────────────────
        if self.use_water_mask:
            wm = np.load(self.wm_dir / fname).astype(np.float32)
            sample["water_mask"] = torch.from_numpy(wm).float()[None]

        # ── TIME METADATA (for time-aware gradient loss) ──────
        # Loaded from metadata.json, not fed into the model.
        # Used only in combined_loss to modulate lambda_grad.
        # Fallback values used if tile is missing from metadata
        # (should not happen — all tiles are in metadata.json).
        key  = Path(fname).stem   # tile name without .npy
        meta = self.metadata.get(key, {})

        sample["lr_time"] = torch.tensor(
            float(meta.get("lr_time", 10.0)),
            dtype=torch.float32,
        )
        sample["time_gap_hours"] = torch.tensor(
            float(meta.get("time_gap_hours", 0.0)),
            dtype=torch.float32,
        )
        sample["date_gap_days"] = torch.tensor(
            float(meta.get("date_gap_days", 0)),
            dtype=torch.float32,
        )

        return sample