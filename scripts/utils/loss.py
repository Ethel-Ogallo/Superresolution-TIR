# scripts/utils/loss.py
"""
loss.py — Loss functions for TIR Super-Resolution.
"""
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient


# ----------------- Loss functions ----------------

def masked_l1(sr: torch.Tensor,
              hr: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    """
    L1 loss on valid pixels only.
    Measures average absolute temperature error — pixel-wise accuracy.
    Does not penalize over-smoothing.

    Args:
        sr, hr : (B, 1, H, W) — denormalised predictions and targets
        mask   : (B, 1, H, W) — 1 valid, 0 nodata
    """
    err = torch.abs(sr - hr) * mask
    return err.sum() / torch.clamp(mask.sum(), min=1.0)


_spatial_gradient = SpatialGradient()


def gradient_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """
    Spatial temperature gradient loss.
    Penalizes differences in spatial gradients between SR and HR.
    Pushes model to recover sharp thermal boundaries (river edges,
    roads, urban structures) that L1 alone misses.

    Uses kornia.filters.SpatialGradient — output shape (B, 1, 2, H, W).
    Border pixels excluded via inner mask to avoid artifacts at nodata edges.

    Args:
        sr, hr : (B, 1, H, W) — denormalised predictions and targets
        mask   : (B, 1, H, W) — 1 valid, 0 nodata
    """
    sr_grads = _spatial_gradient(sr)
    hr_grads = _spatial_gradient(hr)

    mask_inner = (
        F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) > 0.99
    ).float()

    mask_5d = mask_inner.unsqueeze(2).expand_as(sr_grads)
    n = torch.clamp(mask_5d.sum(), min=1.0)
    loss = (torch.abs(sr_grads - hr_grads) * mask_5d).sum() / n

    return loss


def water_aware_loss(sr: torch.Tensor,
                     hr: torch.Tensor,
                     hr_mask: torch.Tensor,
                     water_mask: torch.Tensor) -> torch.Tensor:
    """
    Water-prioritized L1 loss.
    Same as masked_l1 but restricted to valid water pixels only.
    Added on top of base loss with higher weight to focus the model
    on accurate river temperature reconstruction.

    Args:
        sr, hr    : (B, 1, H, W) — denormalised predictions and targets
        hr_mask   : (B, 1, H, W) — 1 valid, 0 nodata
        water_mask: (B, 1, H, W) — 1 water, 0 non-water
    """
    combined_mask = hr_mask * water_mask
    err = torch.abs(sr - hr) * combined_mask
    return err.sum() / torch.clamp(combined_mask.sum(), min=1.0)


def combined_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  hr_mask: torch.Tensor,
                  water_mask: torch.Tensor = None,
                  lambda_grad: float = 0.1,
                  lambda_water: float = 0.5) -> torch.Tensor:
    """
    Combined loss for EXP-03:
        total = masked_l1
              + lambda_grad  * gradient_loss
              + lambda_water * water_aware_loss  (only if water_mask provided)

    Args:
        sr, hr       : (B, 1, H, W) — denormalised predictions and targets
        hr_mask      : (B, 1, H, W) — 1 valid, 0 nodata
        water_mask   : (B, 1, H, W) — 1 water, 0 non-water (optional)
        lambda_grad  : weight for gradient loss (default 0.1)
        lambda_water : weight for water loss (default 0.5)
    """
    loss = masked_l1(sr, hr, hr_mask)
    loss = loss + lambda_grad * gradient_loss(sr, hr, hr_mask)

    if water_mask is not None:
        loss = loss + lambda_water * water_aware_loss(sr, hr, hr_mask, water_mask)

    return loss