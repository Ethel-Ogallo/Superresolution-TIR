"""
loss.py — TIR Super-Resolution Loss (water-prioritized version)
"""
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient


# ----------------- Base L1 -----------------

def masked_l1(sr: torch.Tensor,
              hr: torch.Tensor,
              mask: torch.Tensor) -> torch.Tensor:
    err = torch.abs(sr - hr) * mask
    return err.sum() / torch.clamp(mask.sum(), min=1.0)


# ----------------- Gradient loss -----------------

_spatial_gradient = SpatialGradient()

def gradient_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:

    sr_g = _spatial_gradient(sr)
    hr_g = _spatial_gradient(hr)

    mask_inner = (F.avg_pool2d(mask, 3, 1, 1) > 0.5).float()
    mask_5d = mask_inner.unsqueeze(2).expand_as(sr_g)

    err = torch.abs(sr_g - hr_g) * mask_5d
    return err.sum() / torch.clamp(mask_5d.sum(), min=1.0)


# ----------------- Water-weighted loss -----------------

def water_weighted_loss(sr: torch.Tensor,
                        hr: torch.Tensor,
                        hr_mask: torch.Tensor,
                        water_mask: torch.Tensor,
                        water_weight: float = 2.0,
                        land_weight: float = 1.0) -> torch.Tensor:

    error = torch.abs(sr - hr)

    water_mask = (hr_mask * water_mask).float()
    land_mask  = (hr_mask * (1.0 - water_mask)).float()

    loss_water = (error * water_mask).sum() / torch.clamp(water_mask.sum(), min=1.0)
    loss_land  = (error * land_mask).sum() / torch.clamp(land_mask.sum(), min=1.0)

    return water_weight * loss_water + land_weight * loss_land


# ----------------- FINAL COMBINED LOSS -----------------

def combined_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  hr_mask: torch.Tensor,
                  water_mask: torch.Tensor = None,
                  lambda_grad: float = 0.1,
                  lambda_water: float = 0.5,
                  water_weight: float = 2.0,
                  land_weight: float = 1.0) -> torch.Tensor:

    loss = masked_l1(sr, hr, hr_mask)

    loss = loss + lambda_grad * gradient_loss(sr, hr, hr_mask)

    if water_mask is not None:
        loss = loss + lambda_water * water_weighted_loss(
            sr, hr, hr_mask, water_mask,
            water_weight=water_weight,
            land_weight=land_weight
        )

    return loss