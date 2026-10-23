"""
dataset.py — SR Dataset for TIR Super-Resolution

Loads per-patch data for a 4× SR task on thermal infrared (TIR) imagery
of the Rhône river system.

What each field contains:
lr          [1, 64,  64]   LR TIR patch, z-score normalised
hr          [1, 256, 256]  HR TIR patch, z-score normalised
hr_mask     [1, 256, 256]  1 where HR pixel is valid (not NaN), 0 elsewhere
aux_lr      [20, 64,  64]  Aux data at LR resolution  — direct model input
aux_mid     [20, 128, 128] Aux data at 128px          — SPADE mid conditioning
aux_hr      [20, 256, 256] Aux data at 256px          — SPADE HR conditioning
water_mask  [1,  256, 256] 1 where pixel is water/river 

Time fields (scalars, used in model input AND loss):
time_gap_hours   float   |hr_time  - lr_time|  in hours
date_gap_days    float   |hr_date  - lr_date|  in calendar days

Time fields are loaded as tensors but used in two ways:
    1. Model input  : tiled spatially and concatenated as channels 22-24
                      (time_gap_hours/24, date_gap_days/365)
    2. Loss weight  : passed as scalars to combined_loss to modulate
                      the global loss weight.

Aux channel layout (23ch, fixed at patch creation):
[0:5]   spectral bands 1–5    continuous  [0, 1]
[5]     NDVI                  continuous  [-1, 1]
[6]     NDWI                  continuous  [-1, 1]
[7]     NDMI                  continuous  [-1, 1]
[8:22]  COSIA one-hot (14cls) binary      {0, 1}
[22]    DEM                   continuous  [0, 1]
"""


import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path


class SRDataset(Dataset):
    def __init__(
        self,
        split: str,
        patches_dir: str,
        stats_path: str,
        use_water_mask: bool = True,
        use_aux: bool        = True,
        aux_dir: str         = None,
        transform            = None,
    ):
        """
        Args:
            split        : "train", "val", or "test"
            patches_dir  : root directory containing split subdirs + metadata.json
            stats_path   : path to stats.json (mean/std for LR and HR)
            use_water_mask: whether to load water masks
            use_aux      : whether to load aux channels
            aux_dir      : path to LR-resolution aux folder.
                           Mid (128px) and HR (256px) aux are expected as
                           siblings named AUX_SPADE_MID and AUX_SPADE_HR.
            transform    : optional augmentation callable
        """
        self.split          = split
        self.patches_dir    = Path(patches_dir)
        self.transform      = transform
        self.use_water_mask = use_water_mask
        self.use_aux        = use_aux

        # ── Aux directory layout ──────────────────────────────────────────────
        # aux_lr  — same resolution as LR input (64px)   — fed into model input
        # aux_mid — 128px                                — SPADE mid stage
        # aux_hr  — 256px                                — SPADE HR stage
        if aux_dir is not None:
            base = Path(aux_dir)
            self.aux_lr_dir  = base
            self.aux_mid_dir = base.parent / "AUX_SPADE_MID"
            self.aux_hr_dir  = base.parent / "AUX_SPADE_HR"
        else:
            self.aux_lr_dir  = None
            self.aux_mid_dir = None
            self.aux_hr_dir  = None

        # ── Normalisation statistics ──────────────────────────────────────────
        with open(stats_path) as f:
            stats = json.load(f)
        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        # ── Per-patch time metadata ───────────────────────────────────────────
        # metadata.json stores lr_time, time_gap_hours, date_gap_days per tile.
        # Loaded once at init — accessed per sample in __getitem__.
        metadata_path = self.patches_dir / "metadata.json"
        with open(metadata_path) as f:
            self.metadata = json.load(f)

        # ── Patch file list ───────────────────────────────────────────────────
        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"
        self.wm_dir = self.patches_dir / split / "WM"
        self.files  = sorted([f.name for f in self.hr_dir.glob("*.npy")])

    def __len__(self) -> int:
        return len(self.files)

    def __getitem__(self, idx: int) -> dict:
            fname = self.files[idx]

            # 1. ── Load and Prep LR / HR patches ──────────────────────────────────
            lr = np.load(self.lr_dir / fname).astype(np.float32)
            hr = np.load(self.hr_dir / fname).astype(np.float32)

            hr_mask = np.isfinite(hr).astype(np.float32)
            lr_mask = np.isfinite(lr).astype(np.float32)

            hr = np.where(hr_mask, hr, self.hr_mean)
            lr = np.where(lr_mask, lr, self.lr_mean)

            hr = (hr - self.hr_mean) / self.hr_std
            lr = (lr - self.lr_mean) / self.lr_std

            # 2. ── Load Aux data ──────────────────────────────────────────────────
            aux_lr  = None
            aux_mid = None
            aux_hr  = None
            if self.use_aux and self.aux_lr_dir is not None:
                aux_lr  = np.load(self.aux_lr_dir  / fname).astype(np.float32)
                aux_mid = np.load(self.aux_mid_dir / fname).astype(np.float32)
                aux_hr  = np.load(self.aux_hr_dir  / fname).astype(np.float32)

            # 3. ── Load Water Mask ────────────────────────────────────────────────
            wm = np.load(self.wm_dir / fname).astype(np.float32)

            # 4. ── Augmentation  ──
            if self.transform:
                lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, wm = self.transform(
                    lr, hr, hr_mask, aux_lr, aux_mid, aux_hr, wm
                )

            # 5. ── Convert EVERYTHING to Tensors & Build Sample Dict ─────────────
            sample = {
                "lr":                 torch.from_numpy(lr).float()[None],        # [1, 64,  64]
                "hr":                 torch.from_numpy(hr).float()[None],        # [1, 256, 256]
                "hr_mask":            torch.from_numpy(hr_mask).float()[None],   # [1, 256, 256]
                "water_mask":         torch.from_numpy(wm).float()[None],        # [1, 256, 256]
                "fname":              fname,
            }

            if aux_lr is not None:
                sample["aux_lr"]  = torch.from_numpy(aux_lr).float()   # [20, 64,  64]
                sample["aux_mid"] = torch.from_numpy(aux_mid).float()  # [20, 128, 128]
                sample["aux_hr"]  = torch.from_numpy(aux_hr).float()   # [20, 256, 256]

            # 6. ── Time metadata ─────────────────────────────────────────────────
            key  = Path(fname).stem
            meta = self.metadata.get(key, {})

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