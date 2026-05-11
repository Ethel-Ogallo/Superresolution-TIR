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
    Uses kornia.filters.SpatialGradient — output shape (B, 1, 2, H, W).
    Only valid pixels contribute. Border pixels excluded to avoid
    artificial gradients at nodata edges.
    """
    sr_grads = _spatial_gradient(sr)
    hr_grads = _spatial_gradient(hr)

    mask_inner = (
        F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) > 0.99
    ).float()
    mask_5d = mask_inner.unsqueeze(2).expand_as(sr_grads)

    n    = torch.clamp(mask_5d.sum(), min=1.0)
    loss = (torch.abs(sr_grads - hr_grads) * mask_5d).sum() / n
    return loss