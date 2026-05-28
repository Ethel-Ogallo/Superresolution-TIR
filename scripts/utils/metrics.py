# scripts/utils/metrics.py

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

from scripts.utils.loss import masked_l1, combined_loss, gradient_loss, water_grad_loss


# ---------------- Masks ----------------

def build_masks(hr_mask, water_mask):
    full = hr_mask

    if water_mask is None:
        return full, None, None

    water    = hr_mask * water_mask
    nonwater = hr_mask * (1.0 - water_mask)

    return full, water, nonwater


# ---------------- Metric computation ----------------

def _compute_single_metric_set(sr, hr, mask, module, prefix, sync_dist):

    mask = mask > 0.5

    if mask.sum() == 0:
        for m in ["psnr", "ssim", "mae", "rmse"]:
            module.log(f"{prefix}_{m}", torch.tensor(0.0, device=sr.device),
                       on_step=False, on_epoch=True, sync_dist=sync_dist)
        return

    sr_valid = sr[mask]
    hr_valid = hr[mask]

    mae  = torch.abs(sr_valid - hr_valid).mean()
    rmse = torch.sqrt(((sr_valid - hr_valid) ** 2).mean())
    mse  = ((sr_valid - hr_valid) ** 2).mean()
    psnr = 10 * torch.log10((module.DATA_RANGE ** 2) / (mse + 1e-8))

    # --- SSIM: per-sample, normalised to [0,1], minimum size guard ---
    MIN_SSIM_SIZE = 11
    ssim_fn       = StructuralSimilarityIndexMeasure(data_range=1.0).to(sr.device)
    ssim_scores   = []

    for b in range(sr.shape[0]):
        valid_b = mask[b, 0] if mask.dim() == 4 else mask[b]

        if valid_b.sum() == 0:
            continue

        coords = torch.where(valid_b)
        ymin, ymax = coords[0].min().item(), coords[0].max().item()
        xmin, xmax = coords[1].min().item(), coords[1].max().item()

        if (ymax - ymin + 1) < MIN_SSIM_SIZE or (xmax - xmin + 1) < MIN_SSIM_SIZE:
            continue

        sr_crop = sr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
        hr_crop = hr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]

        # normalise to [0,1] using physical range before SSIM
        # DATA_MIN = p1 (~20.76°C), DATA_RANGE = p99-p1 (~27.59°C)
        sr_norm = (sr_crop - module.DATA_MIN) / module.DATA_RANGE
        hr_norm = (hr_crop - module.DATA_MIN) / module.DATA_RANGE
        sr_norm = sr_norm.clamp(0, 1)
        hr_norm = hr_norm.clamp(0, 1)

        ssim_scores.append(ssim_fn(sr_norm, hr_norm))

    ssim = torch.stack(ssim_scores).mean() if ssim_scores \
        else torch.tensor(0.0, device=sr.device)

    module.log(f"{prefix}_psnr", psnr, on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_ssim", ssim, on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_mae",  mae,  on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True, sync_dist=sync_dist)


# ---------------- Main ----------------

def compute_metrics(module, sr, hr, hr_mask, stage, water_mask=None):

    sync_dist = (stage != "train")

    sr = torch.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
    hr = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

    full_mask, water_mask_out, nonwater_mask = build_masks(hr_mask, water_mask)

    _compute_single_metric_set(sr, hr, full_mask,
                               module, f"{stage}_full", sync_dist)

    if water_mask_out is not None:
        _compute_single_metric_set(sr, hr, water_mask_out,
                                   module, f"{stage}_water", sync_dist)

        _compute_single_metric_set(sr, hr, nonwater_mask,
                                   module, f"{stage}_nonwater", sync_dist)


# ---------------- Shared step ----------------

def shared_step(module, batch, stage: str):

    lr         = batch["lr"]
    hr         = batch["hr"]
    hr_mask    = batch["hr_mask"]
    water_mask = batch.get("water_mask", None)

    if water_mask is not None:
        water_mask = water_mask.to(hr_mask.device)

    # ── Forward pass ─────────────────────────────────────
    sr_img = module(batch)

    # ── Denormalize for loss + metrics ───────────────────
    # Loss is computed in physical units (not normalized space)
    # so gradients reflect real temperature differences
    sr = module.denormalize(sr_img[:, 0:1], module.hr_mean, module.hr_std)
    hr = module.denormalize(hr[:, 0:1],     module.hr_mean, module.hr_std)

    # ── Loss ─────────────────────────────────────────────
    loss_dict = combined_loss(
        sr           = sr,
        hr           = hr,
        hr_mask      = hr_mask,
        water_mask   = water_mask,
        lambda_grad  = module.lambda_grad,
        lambda_water = module.lambda_water,
    )

    # ── Log each component separately to wandb ───────────
    module.log(f"{stage}_l1", loss_dict["l1"],
               on_epoch=True)
    module.log(f"{stage}_loss_grad",  loss_dict["loss_grad"],
               on_epoch=True)
    module.log(f"{stage}_loss_water", loss_dict["loss_water"],
               on_epoch=True)
    module.log(f"{stage}_loss",       loss_dict["loss_total"],
               on_step=(stage == "train"), on_epoch=True, prog_bar=True)

    # ── Metrics ──────────────────────────────────────────
    with torch.no_grad():
        compute_metrics(module, sr, hr, hr_mask, stage, water_mask)

    return loss_dict["loss_total"]