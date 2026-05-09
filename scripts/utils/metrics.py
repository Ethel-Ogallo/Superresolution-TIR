# scripts/utils/metrics.py
"""
metrics.py — Shared metrics and step logic for all SR models.

- full metrics (all valid pixels)
- water-only metrics
- non-water metrics
"""

import torch
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from scripts.utils.loss import masked_l1


# -----------------------------
# MASK HELPERS
# -----------------------------
def build_masks(hr_mask, water_mask):
    """
    Build consistent evaluation masks.
    """
    full = hr_mask

    if water_mask is None:
        return full, None, None

    water = hr_mask * water_mask
    nonwater = hr_mask * (1.0 - water_mask)

    return full, water, nonwater


# -----------------------------
# IMAGE CLEANING
# -----------------------------
def fill_invalid(sr, hr, mask):
    """
    Replace invalid pixels (mask==0) with mean of valid HR pixels.
    """
    sr_out = sr.float().clone()
    hr_out = hr.float().clone()

    for b in range(hr.shape[0]):
        valid = hr_out[b][mask[b] > 0.5]

        fill = valid.mean() if valid.numel() > 0 else torch.tensor(0.0, device=hr.device)
        if not torch.isfinite(fill):
            fill = torch.tensor(0.0, device=hr.device)

        inv = mask[b] <= 0.5
        sr_out[b][inv] = fill
        hr_out[b][inv] = fill

    return sr_out, hr_out


# -----------------------------
# SINGLE METRIC BLOCK
# -----------------------------
def _compute_single_metric_set(sr, hr, mask, module, prefix, sync_dist):
    """
    Computes PSNR / SSIM / MAE / RMSE for a given mask.
    """

    valid = mask > 0.5

    if valid.sum() == 0:
        for m in ["psnr", "ssim", "mae", "rmse"]:
            module.log(f"{prefix}_{m}",
                       torch.tensor(0.0, device=sr.device),
                       on_step=False, on_epoch=True,
                       sync_dist=sync_dist)
        return

    err = (sr - hr)[valid]
    mae = err.abs().mean()
    rmse = err.pow(2).mean().sqrt()

    sr_f, hr_f = fill_invalid(sr, hr, mask)

    psnr_fn = PeakSignalNoiseRatio(data_range=module.DATA_RANGE).to(sr.device)
    psnr = psnr_fn(sr_f, hr_f)

    if not torch.isfinite(psnr):
        psnr = torch.tensor(0.0, device=sr.device)

    if min(sr.shape[-2], sr.shape[-1]) >= 11:
        ssim_fn = StructuralSimilarityIndexMeasure(
            data_range=module.DATA_RANGE
        ).to(sr.device)

        ssim = ssim_fn(
            sr_f.clamp(0, module.DATA_RANGE),
            hr_f.clamp(0, module.DATA_RANGE),
        )

        if not torch.isfinite(ssim):
            ssim = torch.tensor(0.0, device=sr.device)
    else:
        ssim = torch.tensor(0.0, device=sr.device)

    module.log(f"{prefix}_psnr", psnr, on_step=False, on_epoch=True,
               prog_bar=(prefix == "full"), sync_dist=sync_dist)

    module.log(f"{prefix}_ssim", ssim, on_step=False, on_epoch=True,
               sync_dist=sync_dist)

    module.log(f"{prefix}_mae", mae, on_step=False, on_epoch=True,
               prog_bar=(prefix == "full"), sync_dist=sync_dist)

    module.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True,
               sync_dist=sync_dist)


# -----------------------------
# MAIN METRICS FUNCTION
# -----------------------------
def compute_metrics(module, sr, hr, hr_mask, stage, water_mask=None):
    """
    Computes:
    - full valid pixels
    - water pixels
    - non-water pixels
    """

    is_train = stage == "train"
    sync_dist = not is_train

    sr = torch.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
    hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

    full_mask, water_mask_out, nonwater_mask = build_masks(hr_mask, water_mask)

    # FULL
    _compute_single_metric_set(sr, hr, full_mask,
                               module, f"{stage}_full", sync_dist)

    # WATER
    if water_mask_out is not None:
        _compute_single_metric_set(sr, hr, water_mask_out,
                                   module, f"{stage}_water", sync_dist)

        # NON-WATER
        _compute_single_metric_set(sr, hr, nonwater_mask,
                                   module, f"{stage}_nonwater", sync_dist)


# -----------------------------
# TRAIN/VAL/TEST STEP
# -----------------------------
def shared_step(module, batch, stage: str):
    """
    Shared step for all SR models.
    """

    lr = batch["lr"]
    hr = batch["hr"]
    hr_mask = batch["hr_mask"]
    water_mask = batch.get("water_mask", None)

    if water_mask is not None:
        water_mask = water_mask.to(hr_mask.device)

    sr_img = module(lr)

    sr = module.denormalize(sr_img[:, 0:1])
    hr = module.denormalize(hr[:, 0:1])

    loss = masked_l1(sr, hr, hr_mask)

    module.log(f"{stage}_loss", loss,
               on_step=(stage == "train"),
               on_epoch=True,
               prog_bar=True)

    with torch.no_grad():
        compute_metrics(module, sr, hr, hr_mask, stage, water_mask)

    return loss