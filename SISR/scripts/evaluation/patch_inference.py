"""
inference_test_set.py — run the trained SPADE-guided model on every test
patch, compute per-patch water/land metrics, and save as a DataFrame.
"""

import sys, json, time, re
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
CKPT_DIR    = BASE / "checkpoints/gan_final"
OUT_DIR     = BASE / "results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

sys.path.insert(0, str(BASE))

from scripts.utils.dataset import SRDataset
from scripts.utils.metrics import compute_metrics
from scripts.models.realesrgan import RealESRGANModule

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Device: {device}")

# ── Stats + checkpoint ────────────────────────────────────────────────
with open(STATS_PATH) as f:
    stats = json.load(f)

common_kwargs = dict(
    hr_mean=stats["hr"]["mean"],
    hr_std=stats["hr"]["std"],
    data_range=stats["hr_data_range"],
    data_min=stats["hr_percentiles"]["p1"],
)

CKPT_PATH = CKPT_DIR / "exp2_fixed_lr_schedule_epoch=56_val_water_mae=0.8773.ckpt"  

model = RealESRGANModule.load_from_checkpoint(
    str(CKPT_PATH), strict=False, **common_kwargs
)
model.eval().to(device)
print(f"Loaded model | use_spade={model.use_spade}")

# ── Test dataset ───────────────────────────────────────────────────────
test_ds = SRDataset(
    split="test",
    patches_dir=PATCHES_DIR,
    stats_path=STATS_PATH,
    use_aux=True,
    use_water_mask=True,
    aux_dir=str(PATCHES_DIR / "test" / "AUX"),
    transform=None,
)
# batch_size=1 is required here — compute_metrics pools the mask across the
# whole batch dimension, so a batch > 1 would blend metrics across patches
# instead of giving one row per patch.
test_loader = DataLoader(test_ds, batch_size=1, shuffle=False, num_workers=4)
print(f"Test patches: {len(test_ds)}")

def get_campaign(fname):
    m = re.match(r"^([A-Z]+_\d{4})", fname)
    return m.group(1) if m else "unknown"

# ── Inference loop ───────────────────────────────────────────────────────
records = []
t0 = time.time()

with torch.no_grad():
    for batch_idx, batch in enumerate(test_loader):
        fname = batch["fname"][0]  # collated as a length-1 list

        # move everything except the fname string to device
        batch = {k: (v.to(device) if torch.is_tensor(v) else v)
                  for k, v in batch.items()}

        sr = model(batch)                              # normalized output
        hr = batch["hr"]

        sr_phys = model.denormalize(sr)                 # -> °C
        hr_phys = model.denormalize(hr[:, 0:1])

        metrics = compute_metrics(
            module=model,
            sr=sr_phys,
            hr=hr_phys,
            hr_mask=batch["hr_mask"],
            stage="test",
            water_mask=batch.get("water_mask"),
        )

        row = {"fname": fname, "campaign": get_campaign(fname)}
        for k, v in metrics.items():
            row[k] = v.item() if torch.is_tensor(v) else v
        records.append(row)

        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(test_ds):
            print(f"[{batch_idx + 1}/{len(test_ds)}] {fname}")

df_test_metrics = pd.DataFrame(records)
elapsed = (time.time() - t0) / 60
print(f"\nDone in {elapsed:.2f} min | {len(df_test_metrics)} patches, "
      f"{df_test_metrics['campaign'].nunique()} campaigns")
print(df_test_metrics.groupby("campaign").size())

# ── Save ───────────────────────────────────────────────────────────────
out_path = OUT_DIR / "spade_test_patch_metrics.csv"
df_test_metrics.to_csv(out_path, index=False)
print(f"Saved: {out_path}")

