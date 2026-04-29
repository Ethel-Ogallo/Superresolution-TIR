"""
Shared training step — loss, metrics, logging for all SR models.
- Computes masked L1 loss and optional gradient loss.
- Crops to valid bounding box for PSNR/SSIM computation.
- Logs all metrics to W&B via Lightning's self.log.
"""

import torch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scripts.utils.loss import gradient_loss


def build_metrics(data_range: float, device=None):
    """Instantiate PSNR and SSIM metrics for train/val/test splits."""
    metrics = {}
    for split in ("train", "val", "test"):
        metrics[f"{split}_psnr"] = PeakSignalNoiseRatio(data_range=data_range)
        metrics[f"{split}_ssim"] = StructuralSimilarityIndexMeasure(data_range=data_range)
    return metrics


def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
    """Crop to tight bounding box of valid (mask=1) pixels."""
    m    = mask[:, 0, :, :]
    flat = m.any(dim=0)
    rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
    cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
    if rows.numel() == 0 or cols.numel() == 0:
        return tensor, mask
    r0, r1 = rows[0].item(), rows[-1].item() + 1
    c0, c1 = cols[0].item(), cols[-1].item() + 1
    return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]


def shared_step(
    module,
    batch,
    stage: str,
):
    """
    Shared training/validation/test step for all SR models.
    
    Args:
        module:  the LightningModule (has denormalize, hparams, log)
        batch:   (lr, hr, mask, aux) — aux may be zeros if use_aux=False
        stage:   'train' | 'val' | 'test'
    
    Returns:
        loss scalar
    """
    lr_img, hr_img, hr_mask, aux = batch
    sr_img = module(lr_img, aux if module.use_aux else None)

    sr = module.denormalize(sr_img)
    hr = module.denormalize(hr_img)

    # Masked L1 reconstruction loss
    abs_err    = torch.abs(sr - hr) * hr_mask
    recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

    # Gradient loss
    if module.hparams.lambda_grad > 0:
        grad_loss = gradient_loss(sr, hr, hr_mask)
        loss = recon_loss + module.hparams.lambda_grad * grad_loss
    else:
        grad_loss = torch.tensor(0.0, device=sr.device)
        loss = recon_loss

    # Metrics
    with torch.no_grad():
        sr_crop, mask_crop = crop_to_valid_bbox(sr, hr_mask)
        hr_crop, _         = crop_to_valid_bbox(hr, hr_mask)

        sr_crop = torch.nan_to_num(sr_crop, nan=0.0, posinf=0.0, neginf=0.0)
        hr_crop = torch.nan_to_num(hr_crop, nan=0.0, posinf=0.0, neginf=0.0)

        valid = (mask_crop > 0.5)

        if valid.sum() == 0:
            psnr_val = torch.tensor(0.0, device=sr.device)
            ssim_val = torch.tensor(0.0, device=sr.device)
            mae_val  = torch.tensor(0.0, device=sr.device)
        else:
            # PSNR
            sr_valid = sr_crop.clone()
            hr_valid = hr_crop.clone()
            sr_valid[~valid] = 0
            hr_valid[~valid] = 0
            psnr_val = getattr(module, f"{stage}_psnr")(sr_valid, hr_valid)

            # SSIM
            h, w = sr_crop.shape[-2], sr_crop.shape[-1]
            if min(h, w) < 11:
                ssim_val = torch.tensor(0.0, device=sr.device)
            else:
                sr_for_ssim = sr_crop.clone()
                hr_for_ssim = hr_crop.clone()
                sr_for_ssim[~valid] = 0
                hr_for_ssim[~valid] = 0
                ssim_val = getattr(module, f"{stage}_ssim")(sr_for_ssim, hr_for_ssim)

            # MAE
            mae_val = (sr_crop - hr_crop)[valid].abs().mean()

    module.log(f"{stage}_loss",       loss,       on_epoch=True, prog_bar=True)
    module.log(f"{stage}_recon_loss", recon_loss, on_epoch=True, prog_bar=False)
    module.log(f"{stage}_grad_loss",  grad_loss,  on_epoch=True, prog_bar=False)
    module.log(f"{stage}_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
    module.log(f"{stage}_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
    module.log(f"{stage}_mae",        mae_val,    on_epoch=True, prog_bar=True)

    return loss