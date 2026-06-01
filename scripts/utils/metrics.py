# scripts/utils/metrics.py

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure
from scripts.utils.loss import masked_l1, combined_loss, gradient_loss, water_grad_loss


# =========================================================
# MASK HELPERS
# =========================================================

def build_masks(hr_mask, water_mask):
    full = hr_mask
    if water_mask is None:
        return full, None, None
    water    = hr_mask * water_mask
    nonwater = hr_mask * (1.0 - water_mask)
    return full, water, nonwater


# =========================================================
# SINGLE METRIC SET
# =========================================================

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

    # SSIM: per-sample, normalised to [0,1], minimum size guard
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

        sr_norm = ((sr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)
        hr_norm = ((hr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)

        ssim_scores.append(ssim_fn(sr_norm, hr_norm))

    ssim = torch.stack(ssim_scores).mean() if ssim_scores \
        else torch.tensor(0.0, device=sr.device)

    module.log(f"{prefix}_psnr", psnr, on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_ssim", ssim, on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_mae",  mae,  on_step=False, on_epoch=True, sync_dist=sync_dist)
    module.log(f"{prefix}_rmse", rmse, on_step=False, on_epoch=True, sync_dist=sync_dist)


# =========================================================
# COMPUTE METRICS
# =========================================================

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


# =========================================================
# SHARED STEP
# =========================================================

def shared_step(module, batch, stage: str):

    hr_mask    = batch["hr_mask"]
    water_mask = batch.get("water_mask", None)

    if water_mask is not None:
        water_mask = water_mask.to(hr_mask.device)

    # Time metadata for time-aware gradient loss
    # .mean().item() reduces batch dim to scalar for loss functions
    lr_time        = batch["lr_time"].mean().item()
    time_gap_hours = batch["time_gap_hours"].mean().item()
    date_gap_days  = batch["date_gap_days"].mean().item()

    # Forward pass
    sr_img = module(batch)

    # Denormalize — loss and metrics computed in physical units
    # so values reflect real temperature differences
    sr = module.denormalize(sr_img[:, 0:1], module.hr_mean, module.hr_std)
    hr = module.denormalize(
        batch["hr"][:, 0:1], module.hr_mean, module.hr_std
    )

    # Loss
    loss_dict = combined_loss(
        sr             = sr,
        hr             = hr,
        hr_mask        = hr_mask,
        water_mask     = water_mask,
        lambda_grad    = module.lambda_grad,
        lambda_water   = module.lambda_water,
        lr_time        = lr_time,
        time_gap_hours = time_gap_hours,
        date_gap_days  = date_gap_days,
    )

    b_size = hr.shape[0]

    # Log individual components for diagnosis in wandb
    module.log(f"{stage}_l1",
               loss_dict["l1"].detach(),
               on_epoch=True, batch_size=b_size)
    module.log(f"{stage}_loss_grad",
               loss_dict["loss_grad"].detach(),
               on_epoch=True, batch_size=b_size)
    module.log(f"{stage}_loss_grad_weight",
               loss_dict["loss_grad_weight"].detach(),
               on_epoch=True, batch_size=b_size)
    module.log(f"{stage}_loss_water",
               loss_dict["loss_water"].detach(),
               on_epoch=True, batch_size=b_size)
    module.log(f"{stage}_loss",
               loss_dict["loss_total"],
               on_step=(stage == "train"),
               on_epoch=True,
               prog_bar=True,
               batch_size=b_size)

    with torch.no_grad():
        compute_metrics(module, sr, hr, hr_mask, stage, water_mask)

    return loss_dict["loss_total"]