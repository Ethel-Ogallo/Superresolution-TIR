# eval_ssim.py
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from torchmetrics.image import StructuralSimilarityIndexMeasure

from scripts.utils.dataset import SRDataset
from scripts.models.swinir import SwinIRModule

BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"

with open(STATS_PATH) as f:
    stats = json.load(f)

data_min   = stats["hr_percentiles"]["p1"]    # 20.76
data_range = stats["hr_data_range"]           # 27.59

CHECKPOINTS = {
    "projection": "path/to/projection_best.ckpt",
    "direct":     "path/to/direct_best.ckpt",
    "fusion":     "path/to/fusion_best.ckpt",
}

def compute_ssim_for_ckpt(ckpt_path, strategy):

    model = SwinIRModule.load_from_checkpoint(ckpt_path)
    model.eval()
    model.cuda()

    ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "test" / "AUX"),
        repeat_channels=False,
        transform=None,
    )

    loader   = DataLoader(ds, batch_size=4, shuffle=False, num_workers=4)
    ssim_fn  = StructuralSimilarityIndexMeasure(data_range=1.0).cuda()
    MIN_SIZE = 11
    scores   = []

    with torch.no_grad():
        for batch in loader:
            batch   = {k: v.cuda() if isinstance(v, torch.Tensor) else v
                       for k, v in batch.items()}

            sr_img  = model(batch)
            sr      = model.denormalize(sr_img[:, 0:1])
            hr      = model.denormalize(batch["hr"][:, 0:1])
            hr_mask = batch["hr_mask"]

            for b in range(sr.shape[0]):
                valid_b = hr_mask[b, 0] > 0.5
                if valid_b.sum() == 0:
                    continue

                coords = torch.where(valid_b)
                ymin, ymax = coords[0].min().item(), coords[0].max().item()
                xmin, xmax = coords[1].min().item(), coords[1].max().item()

                if (ymax - ymin + 1) < MIN_SIZE or (xmax - xmin + 1) < MIN_SIZE:
                    continue

                sr_crop = sr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
                hr_crop = hr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]

                sr_norm = ((sr_crop - data_min) / data_range).clamp(0, 1)
                hr_norm = ((hr_crop - data_min) / data_range).clamp(0, 1)

                scores.append(ssim_fn(sr_norm, hr_norm).item())

    mean_ssim = sum(scores) / len(scores) if scores else 0.0
    print(f"{strategy:15s}  SSIM = {mean_ssim:.4f}  (n={len(scores)} patches)")
    return mean_ssim


for strategy, ckpt_path in CHECKPOINTS.items():
    compute_ssim_for_ckpt(ckpt_path, strategy)