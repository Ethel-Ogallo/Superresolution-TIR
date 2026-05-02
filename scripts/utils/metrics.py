"""
metrics.py — Shared training step for all SR models.

Computes:
  - Masked L1 reconstruction loss
  - Optional gradient loss
  - PSNR / SSIM / MAE on the valid bounding box only

PSNR and SSIM are computed only over valid pixels by masking invalid
regions to the valid mean rather than zero — this prevents zero-vs-zero
pairs from inflating scores when nodata borders are large.

FiLM-safe and backbone-agnostic.
"""

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scripts.utils.loss import gradient_loss


def build_metrics(data_range: float):
    metrics = {}
    for split in ("train", "val", "test"):
        metrics[f"{split}_psnr"] = PeakSignalNoiseRatio(data_range=data_range)
        metrics[f"{split}_ssim"] = StructuralSimilarityIndexMeasure(data_range=data_range)
    return metrics


def crop_to_valid_bbox(sr: torch.Tensor, hr: torch.Tensor, mask: torch.Tensor):
    """
    Crop sr, hr, and mask to the tight bounding box of valid pixels.
    Returns originals unchanged if the valid region is too small for SSIM.
    """
    valid = mask[:, 0]                                      # (B, H, W)
    rows  = valid.any(dim=2).any(dim=0).nonzero(as_tuple=True)[0]
    cols  = valid.any(dim=1).any(dim=0).nonzero(as_tuple=True)[0]

    if rows.numel() < 11 or cols.numel() < 11:
        return sr, hr, mask

    r0, r1 = rows[0].item(), rows[-1].item() + 1
    c0, c1 = cols[0].item(), cols[-1].item() + 1

    return (
        sr  [:, :, r0:r1, c0:c1],
        hr  [:, :, r0:r1, c0:c1],
        mask[:, :, r0:r1, c0:c1],
    )


def _fill_invalid(tensor: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """
    Replace invalid pixels (mask == 0) with the per-image valid mean.
    Prevents zero-vs-zero pairs from inflating PSNR/SSIM on nodata borders.
    Falls back to global tensor mean, then 0.0, to guard against NaN.
    """
    out = tensor.clone()
    for b in range(tensor.shape[0]):
        valid_mask = mask[b] > 0.5
        valid_vals = tensor[b][valid_mask]

        if valid_vals.numel() > 0:
            fill = valid_vals.mean()
        else:
            all_vals = tensor[b]
            fill = all_vals.mean() if all_vals.numel() > 0 else torch.tensor(0.0, device=tensor.device)

        if not torch.isfinite(fill):
            fill = torch.tensor(0.0, device=tensor.device)

        out[b] = torch.where(valid_mask, tensor[b], fill)

    return out


def _safe_ssim(module, sr: torch.Tensor, hr: torch.Tensor, stage: str, device) -> torch.Tensor:
    """
    Compute SSIM with full NaN/Inf guards.

    Root cause of Inf: torchmetrics SSIM divides by data_range^2 internally.
    If denormalized values exceed data_range, numerator terms dominate and
    blow up. Fix: clamp both tensors to [0, data_range] before the call.
    """
    # Must be 4D
    if sr.dim() == 3:
        sr = sr.unsqueeze(0)
        hr = hr.unsqueeze(0)

    # Spatial size check
    if min(sr.shape[-2], sr.shape[-1]) < 11:
        return torch.tensor(0.0, device=device)

    # Check inputs are finite
    if not (torch.isfinite(sr).all() and torch.isfinite(hr).all()):
        print(f"[WARNING] SSIM skipped — non-finite inputs "
              f"(sr bad: {(~torch.isfinite(sr)).sum().item()}, "
              f"hr bad: {(~torch.isfinite(hr)).sum().item()})")
        return torch.tensor(0.0, device=device)

    # Clamp to [0, data_range] — prevents Inf when true range exceeds data_range.
    # data_range is computed from the 1st-99th percentile of the training set,
    # so outlier batches can legitimately exceed it after denormalization.
    data_range = module.DATA_RANGE
    sr = sr.clamp(0.0, data_range)
    hr = hr.clamp(0.0, data_range)

    try:
        val = getattr(module, f"{stage}_ssim")(sr, hr)
        if not torch.isfinite(val):
            print(f"[WARNING] SSIM non-finite ({val.item():.4f}) after clamping "
                  f"| shape={sr.shape} | data_range={data_range:.2f} "
                  f"| sr=[{sr.min():.2f}, {sr.max():.2f}] "
                  f"| hr=[{hr.min():.2f}, {hr.max():.2f}]")
            return torch.tensor(0.0, device=device)
        return val
    except RuntimeError as e:
        print(f"[WARNING] SSIM RuntimeError on shape {sr.shape}: {e}")
        return torch.tensor(0.0, device=device)


def shared_step(module, batch, stage: str):
    """
    Shared training/validation/test step for all SR models.

    Args:
        module: LightningModule — must implement forward() and denormalize()
        batch:  (lr, hr, mask, aux)
        stage:  'train' | 'val' | 'test'

    Returns:
        loss scalar
    """
    lr_img, hr_img, hr_mask, aux = batch
    aux_input = aux if (module.use_aux and aux is not None) else None
    sr_img = module(lr_img, aux_input)

    sr = module.denormalize(sr_img)
    hr = module.denormalize(hr_img)

    assert sr.shape == hr.shape, (
        f"Shape mismatch: SR {sr.shape} vs HR {hr.shape}"
    )

    # ---- Masked L1 loss ----
    abs_err    = torch.abs(sr - hr) * hr_mask
    recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

    # ---- Gradient loss ----
    if module.hparams.lambda_grad > 0:
        grad_loss = gradient_loss(sr, hr, hr_mask)
        loss = recon_loss + module.hparams.lambda_grad * grad_loss
    else:
        grad_loss = torch.tensor(0.0, device=sr.device)
        loss = recon_loss

    # ---- Metrics (no grad) ----
    with torch.no_grad():
        sr_clean = torch.nan_to_num(sr, nan=0.0, posinf=0.0, neginf=0.0)
        hr_clean = torch.nan_to_num(hr, nan=0.0, posinf=0.0, neginf=0.0)

        # Crop all three together — single bbox, guaranteed alignment
        sr_crop, hr_crop, mask_crop = crop_to_valid_bbox(sr_clean, hr_clean, hr_mask)

        valid = mask_crop > 0.5

        if valid.sum() == 0 or sr_crop.numel() == 0:
            psnr_val = torch.tensor(0.0, device=sr.device)
            ssim_val = torch.tensor(0.0, device=sr.device)
            mae_val  = torch.tensor(0.0, device=sr.device)
        else:
            # MAE — valid pixels only
            mae_val = (sr_crop - hr_crop)[valid].abs().mean()

            # Fill invalid pixels with per-image valid mean for PSNR/SSIM
            sr_filled = _fill_invalid(sr_crop, mask_crop)
            hr_filled = _fill_invalid(hr_crop, mask_crop)

            # PSNR
            psnr_val = getattr(module, f"{stage}_psnr")(sr_filled, hr_filled)
            if not torch.isfinite(psnr_val):
                print(f"[WARNING] PSNR non-finite ({psnr_val.item():.4f}), replacing with 0.0")
                psnr_val = torch.tensor(0.0, device=sr.device)

            # SSIM — fully guarded, clamped to data_range
            ssim_val = _safe_ssim(module, sr_filled, hr_filled, stage, sr.device)

    # ---- Logging ----
    is_train  = stage == "train"
    sync_dist = not is_train

    module.log(f"{stage}_loss",       loss,       on_step=is_train, on_epoch=True,
               prog_bar=True,  sync_dist=sync_dist)
    module.log(f"{stage}_recon_loss", recon_loss, on_step=False,    on_epoch=True,
               sync_dist=sync_dist)
    module.log(f"{stage}_grad_loss",  grad_loss,  on_step=False,    on_epoch=True,
               sync_dist=sync_dist)
    module.log(f"{stage}_psnr",       psnr_val,   on_step=False,    on_epoch=True,
               prog_bar=True,  sync_dist=sync_dist)
    module.log(f"{stage}_ssim",       ssim_val,   on_step=False,    on_epoch=True,
               prog_bar=True,  sync_dist=sync_dist)
    module.log(f"{stage}_mae",        mae_val,    on_step=False,    on_epoch=True,
               prog_bar=True,  sync_dist=sync_dist)

    return loss