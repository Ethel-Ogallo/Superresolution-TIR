import json
import numpy as np
import torch
import random
from torch.utils.data import Dataset
from pathlib import Path
from scipy.ndimage import gaussian_filter

# --- Data augmentation---
class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    def __call__(self, lr, hr, hr_mask, water_mask):
        for t in self.transforms:
            lr, hr, hr_mask, water_mask = t(lr, hr, hr_mask, water_mask)
        return lr, hr, hr_mask, water_mask

class Flip:
    def __init__(self, p=0.5): self.p = p
    def __call__(self, lr, hr, hr_mask, water_mask):
        if random.random() < self.p:
            axis = random.choice([-1, -2]) # Horizontal or Vertical
            lr, hr, hr_mask, water_mask = [np.flip(x, axis=axis).copy() for x in [lr, hr, hr_mask, water_mask]]
        return lr, hr, hr_mask, water_mask

class Rotation:
    def __call__(self, lr, hr, hr_mask, water_mask):
        k = random.randint(0, 3)
        if k > 0:
            lr, hr, hr_mask, water_mask = [np.rot90(x, k, axes=(-2, -1)).copy() for x in [lr, hr, hr_mask, water_mask]]
        return lr, hr, hr_mask, water_mask

# --- Dataset Class ---

class SRDataset(Dataset):
    def __init__(self, split, processed_dir, stats_path, seq_len=5, augment=False):
        self.split = split
        self.root = Path(processed_dir) / split
        self.seq_len = seq_len
        self.augment = augment
        
        # Define transform chain if augmentation is enabled
        self.transform = Compose([Flip(p=0.5), Rotation()]) if augment else None
        
        with open(stats_path) as f:
            stats = json.load(f)
        self.hr_mean, self.hr_std = stats["hr"]["mean"], stats["hr"]["std"]
        self.lr_mean, self.lr_std = stats["lr"]["mean"], stats["lr"]["std"]
        
        with open(Path(processed_dir) / "metadata.json") as f:
            self.metadata = json.load(f)
            
        self.seq_names = sorted([d.name for d in (self.root / "HR").iterdir() if d.is_dir()])

    def __len__(self):
        return len(self.seq_names)

    def __getitem__(self, idx):
        seq_name = self.seq_names[idx]
        hr_frames, lr_frames, hr_mask_frames, water_mask_frames = [], [], [], []
        for f in range(self.seq_len):
            fname = f"{f:04d}.npy"
            hr = np.load(self.root / "HR" / seq_name / fname).astype(np.float32)
            lr = np.load(self.root / "LR" / seq_name / fname).astype(np.float32)
            water_mask = np.load(self.root / "WM" / seq_name / fname).astype(np.float32)

            hr_mask = np.isfinite(hr).astype(np.float32)  #binary mask: 1 for valid pixels, 0 for NaN
            lr_mask = np.isfinite(lr).astype(np.float32)

            hr = np.where(hr_mask, hr, self.hr_mean)
            lr = np.where(lr_mask, lr, self.lr_mean)

            hr_frames.append((hr - self.hr_mean) / self.hr_std)
            lr_frames.append((lr - self.lr_mean) / self.lr_std)
            hr_mask_frames.append(hr_mask)
            water_mask_frames.append(water_mask)

        lr_seq  = np.stack(lr_frames)
        hr_seq  = np.stack(hr_frames)
        hr_mask_seq    = np.stack(hr_mask_frames)
        water_mask_seq = np.stack(water_mask_frames)

        if self.augment and self.transform:
            lr_seq, hr_seq, hr_mask_seq, water_mask_seq = self.transform(
                lr_seq, hr_seq, hr_mask_seq, water_mask_seq
            )

        lr_tensor = torch.from_numpy(lr_seq).unsqueeze(1).repeat(1, 3, 1, 1)
        hr_tensor = torch.from_numpy(hr_seq).unsqueeze(1).repeat(1, 3, 1, 1)
        hr_mask_tensor    = torch.from_numpy(hr_mask_seq).unsqueeze(1)
        water_mask_tensor = torch.from_numpy(water_mask_seq).unsqueeze(1)

        meta = self.metadata.get(f"{seq_name}_f0000", {})
        return {
            "spatial_key": seq_name,
            "lr": lr_tensor,
            "hr": hr_tensor,
            "hr_mask": hr_mask_tensor,
            "water_mask": water_mask_tensor,
            "time_gap": torch.tensor(float(meta.get("time_gap_hours", 0.0))),
            "date_gap": torch.tensor(float(meta.get("date_gap_days", 0.0)))
        }