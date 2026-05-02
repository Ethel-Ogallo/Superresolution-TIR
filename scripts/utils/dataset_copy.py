"""
dataset.py — TIR Super-Resolution Dataset (Phase 1 baseline)

- Loads pre-tiled HR/LR .npy pairs
- Applies global normalisation using pre-computed stats
- Returns HR mask for NaN pixels (for masked loss in Phase 2)
- Returns LR mask for visualisation
- Repeats single thermal channel 3 times for backbone compatibility
- No random cropping (tiles are pre-cropped)
- No aux data (Phase 1 only)
- Augmentations imported from augmentations.py (Phase 2)
"""

import json
import numpy as np
import torch
from torch.utils.data import Dataset
from pathlib import Path

# ─── Dataset ──────────────────────────────────────────────────────────────────
class SRDataset(Dataset):

    def __init__(
        self,
        split,                          # "train", "val", or "test"
        patches_dir,                    # Path to patches root directory
        stats_path,                     # Path to stats.json
        repeat_channels=True,           # Repeat single channel 3 times
        transform=None,                 # Augmentations (Phase 2 only)
    ):
        self.split           = split
        self.patches_dir     = Path(patches_dir)
        self.repeat_channels = repeat_channels
        self.transform       = transform

        # Load stats
        with open(stats_path) as f:
            stats = json.load(f)
        self.hr_mean = stats["hr"]["mean"]
        self.hr_std  = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std  = stats["lr"]["std"]

        # Build file list
        self.hr_dir = self.patches_dir / split / "HR"
        self.lr_dir = self.patches_dir / split / "LR"

        self.files = sorted([f.name for f in self.hr_dir.glob("*.npy")])

        if len(self.files) == 0:
            raise FileNotFoundError(
                f"[WARN] No .npy files found in {self.hr_dir}"
            )

        # print(f"SRDataset [{split}]: {len(self.files)} tiles loaded")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        fname = self.files[idx]
        hr = np.load(self.hr_dir / fname).astype(np.float32)  # (256, 256)
        lr = np.load(self.lr_dir / fname).astype(np.float32)  # (64,  64)

        # Build masks before any filling
        # True where valid, False where NaN/nodata
        hr_mask = np.isfinite(hr).astype(np.float32)          # (256, 256)
        lr_mask = np.isfinite(lr).astype(np.float32)          # (64,  64)

        # Fill NaN with mean before normalisation. Prevents NaN from propagating through conv layers
        hr = np.where(hr_mask.astype(bool), hr, self.hr_mean)
        lr = np.where(lr_mask.astype(bool), lr, self.lr_mean)

        #  Normalise 
        hr = (hr - self.hr_mean) / self.hr_std
        lr = (lr - self.lr_mean) / self.lr_std

        # Augmentations (Phase 2) 
        if self.transform is not None:
            lr, hr, hr_mask = self.transform(lr, hr, hr_mask)

        # Repeat channels (1 → 3)
        if self.repeat_channels:
            hr = np.stack([hr, hr, hr], axis=0)        # (3, 256, 256)
            lr = np.stack([lr, lr, lr], axis=0)        # (3, 64,  64)
        else:
            hr = hr[np.newaxis]                        # (1, 256, 256)
            lr = lr[np.newaxis]                        # (1, 64,  64)

        # Convert to tensors and add channel dim to masks
        hr      = torch.from_numpy(hr).float()
        lr      = torch.from_numpy(lr).float()
        hr_mask = torch.from_numpy(hr_mask).unsqueeze(0).float()  # (1, 256, 256)
        lr_mask = torch.from_numpy(lr_mask).unsqueeze(0).float()  # (1, 64,  64)

        return {
            "lr":      lr,        # (3, 64,  64)  — model input
            "hr":      hr,        # (3, 256, 256) — ground truth
            "hr_mask": hr_mask,   # (1, 256, 256) — valid pixel mask for loss
            "lr_mask": lr_mask,   # (1, 64,  64)  — valid pixel mask for viz
            "fname":   fname      # tile name for debugging/visualisation
        }


# ─── Quick test ───────────────────────────────────────────────────────────────
# if __name__ == "__main__":
#     PATCHES_DIR = Path("/share/home/e2406751/Superresolution-TIR/data/processed/patches")
#     STATS_PATH  = PATCHES_DIR / "stats.json"

#     for split in ["train", "val", "test"]:
#         ds = SRDataset(
#             split=split,
#             patches_dir=PATCHES_DIR,
#             stats_path=STATS_PATH,
#             repeat_channels=True,
#         )

#         sample = ds[0]
#         print(f"\n{split.upper()} sample:")
#         print(f"  lr      : {sample['lr'].shape}      dtype={sample['lr'].dtype}")
#         print(f"  hr      : {sample['hr'].shape}  dtype={sample['hr'].dtype}")
#         print(f"  hr_mask : {sample['hr_mask'].shape}  dtype={sample['hr_mask'].dtype}")
#         print(f"  lr_mask : {sample['lr_mask'].shape}      dtype={sample['lr_mask'].dtype}")
#         print(f"  fname   : {sample['fname']}")
#         print(f"  lr range: {sample['lr'].min():.3f} – {sample['lr'].max():.3f}")
#         print(f"  hr range: {sample['hr'].min():.3f} – {sample['hr'].max():.3f}")
#         print(f"  hr valid pixels: {sample['hr_mask'].sum().int()} / {256*256}")
#         print(f"  lr valid pixels: {sample['lr_mask'].sum().int()} / {64*64}")
