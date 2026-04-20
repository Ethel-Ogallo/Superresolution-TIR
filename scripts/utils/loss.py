'''
loss.py — Custom loss functions for TIR super-resolution.
'''


import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient

# ----------Gradient loss ---------------
_spatial_gradient = SpatialGradient()  

def gradient_loss(sr: torch.Tensor, hr: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """
    Spatial temperature gradient loss.
    Uses kornia.filters.SpatialGradient - output shape (B, 1, 2, H, W)
    where dim 2 is (x_gradient, y_gradient).
    Only valid (non-nodata) pixels contribute.

    sr, hr, mask : (B, 1, H, W) — in physical units (°C)
    """
    sr_grads = _spatial_gradient(sr)   # (B, 1, 2, H, W)
    hr_grads = _spatial_gradient(hr)   # (B, 1, 2, H, W)

    # Erode mask by 1px — border pixels produce artificial gradients at nodata edges
    mask_inner = (F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) > 0.99).float()
    mask_5d    = mask_inner.unsqueeze(2).expand_as(sr_grads)   # (B, 1, 2, H, W)

    n    = torch.clamp(mask_5d.sum(), min=1.0)
    loss = (torch.abs(sr_grads - hr_grads) * mask_5d).sum() / n
    return loss

