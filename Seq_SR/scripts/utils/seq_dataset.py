import json
import torch
import numpy as np
from torch.utils.data import Dataset
from pathlib import Path
import h5py

class SRSequenceDataset(Dataset):
    def __init__(self, split, patches_dir, stats_path):
        self.split = split
        self.root = Path(patches_dir)
        with open(self.root / "sequence_manifest.json") as f:
            all_sequences = json.load(f)
        self.sequences = [s for s in all_sequences if s["split"] == self.split]
        with open(self.root / "frame_metadata.json") as f:
            self.frame_metadata = json.load(f)

        with open(stats_path) as f:
            stats = json.load(f)
        self.hr_mean = stats["hr"]["mean"]
        self.hr_std = stats["hr"]["std"]
        self.lr_mean = stats["lr"]["mean"]
        self.lr_std = stats["lr"]["std"]

        self.hr_h5 = h5py.File(self.root / "hr_frames.h5", 'r')
        self.lr_h5 = h5py.File(self.root / "lr_frames.h5", 'r')
        self.water_h5 = h5py.File(self.root / "water_masks.h5", 'r')
        self.aux_h5 = h5py.File(self.root / "aux_frames.h5", 'r')
        self.aux_mid_h5 = h5py.File(self.root / "aux_mid_frames.h5", 'r')   # NEW
        self.aux_hr_h5  = h5py.File(self.root / "aux_hr_frames.h5", 'r')    # NEW

        self.hr_data = self.hr_h5['frames']
        self.lr_data = self.lr_h5['frames']
        self.water_data = self.water_h5['frames']
        self.aux_data = self.aux_h5['frames']
        self.aux_mid_data = self.aux_mid_h5['frames']   # NEW  [N, 14, 128, 128]
        self.aux_hr_data  = self.aux_hr_h5['frames']     # NEW  [N, 14, 256, 256]

        print(f"[{split}] Loaded HDF5 files | Sequences: {len(self.sequences)}")

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        seq_entry = self.sequences[idx]
        frame_ids = seq_entry["frame_ids"]

        lr_frames, hr_frames, hr_masks, water_masks, aux_frames = [], [], [], [], []
        aux_mid_frames, aux_hr_frames = [], []   # NEW
        time_gaps, date_gaps = [], []

        for fid in frame_ids:
            frame_idx = int(fid.replace("f", ""))

            lr = self.lr_data[frame_idx].astype(np.float32)
            lr_mask = np.isfinite(lr).astype(np.float32)
            lr = np.where(lr_mask, lr, self.lr_mean)
            lr = (lr - self.lr_mean) / self.lr_std
            lr_frames.append(lr)

            hr = self.hr_data[frame_idx].astype(np.float32)
            water_mask = self.water_data[frame_idx].astype(np.float32)
            hr_mask = np.isfinite(hr).astype(np.float32)
            hr = np.where(hr_mask, hr, self.hr_mean)
            hr = (hr - self.hr_mean) / self.hr_std

            hr_frames.append(hr)
            hr_masks.append(hr_mask)
            water_masks.append(water_mask)

            aux = self.aux_data[frame_idx].astype(np.float32)   # [23, 64, 64]

            meta = self.frame_metadata.get(fid, {})
            time_gap_hours = float(meta.get("time_gap_hours", 0.0))
            date_gap_days = float(meta.get("date_gap_days", 0.0))

            tod_norm = np.clip(abs(time_gap_hours) / 24.0, 0.0, 1.0)
            doy_norm = np.clip(abs(date_gap_days) / 365.0, 0.0, 1.0)

            tod_channel = np.full((aux.shape[1], aux.shape[2]), tod_norm, dtype=np.float32)
            doy_channel = np.full((aux.shape[1], aux.shape[2]), doy_norm, dtype=np.float32)

            aux_with_temporal = np.concatenate([aux, tod_channel[None], doy_channel[None]], axis=0)  # [25, 64, 64]
            aux_frames.append(aux_with_temporal)

            aux_mid_frames.append(self.aux_mid_data[frame_idx].astype(np.float32))  # [14, 128, 128]
            aux_hr_frames.append(self.aux_hr_data[frame_idx].astype(np.float32))    # [14, 256, 256]

            time_gaps.append(time_gap_hours)
            date_gaps.append(date_gap_days)

        lr_seq = np.stack(lr_frames, axis=0)
        hr_seq = np.stack(hr_frames, axis=0)
        hr_mask_seq = np.stack(hr_masks, axis=0)
        water_mask_seq = np.stack(water_masks, axis=0)
        aux_seq = np.stack(aux_frames, axis=0)         # [T, 25, 64, 64]
        aux_mid_seq = np.stack(aux_mid_frames, axis=0)  # [T, 14, 128, 128]
        aux_hr_seq  = np.stack(aux_hr_frames, axis=0)   # [T, 14, 256, 256]

        return {
            "spatial_key": seq_entry["sequence_name"],
            "frame_ids": frame_ids,
            "lr": torch.from_numpy(lr_seq).unsqueeze(1),
            "aux": torch.from_numpy(aux_seq),
            "aux_mid": torch.from_numpy(aux_mid_seq),   
            "aux_hr": torch.from_numpy(aux_hr_seq),     
            "hr": torch.from_numpy(hr_seq).unsqueeze(1),
            "hr_mask": torch.from_numpy(hr_mask_seq).unsqueeze(1),
            "water_mask": torch.from_numpy(water_mask_seq).unsqueeze(1),
            "time_gap": torch.tensor(time_gaps, dtype=torch.float32),
            "date_gap": torch.tensor(date_gaps, dtype=torch.float32),
        }

    def __del__(self):
        for h5 in ("hr_h5", "lr_h5", "water_h5", "aux_h5", "aux_mid_h5", "aux_hr_h5"):  
            if hasattr(self, h5):
                getattr(self, h5).close()