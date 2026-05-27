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


# ----------------- Gradient loss (magnitude) -----------------
# Matches Thomas's grad_loss: sqrt(dx² + dy²) then L1 on magnitudes
_spatial_gradient = SpatialGradient()

def gradient_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:

    sr_g = _spatial_gradient(sr)  # [B, C, 2, H, W]
    hr_g = _spatial_gradient(hr)  # [B, C, 2, H, W]

    # gradient magnitude — matches Thomas's sqrt(gx² + gy²)
    sr_mag = torch.sqrt(sr_g[:, :, 0]**2 + sr_g[:, :, 1]**2 + 1e-8)
    hr_mag = torch.sqrt(hr_g[:, :, 0]**2 + hr_g[:, :, 1]**2 + 1e-8)

    # erode mask slightly to avoid boundary artifacts
    mask_inner = (F.avg_pool2d(mask, 3, 1, 1) > 0.99).float()

    err = torch.abs(sr_mag - hr_mag) * mask_inner
    return err.sum() / torch.clamp(mask_inner.sum(), min=1.0)


# ----------------- Water-weighted loss -----------------
def water_weighted_loss(sr: torch.Tensor,
                        hr: torch.Tensor,
                        hr_mask: torch.Tensor,
                        water_mask: torch.Tensor,
                        water_weight: float = 2.0,
                        land_weight: float = 1.0) -> torch.Tensor:
    error      = torch.abs(sr - hr)
    water_mask = (hr_mask * water_mask).float()
    land_mask  = (hr_mask * (1.0 - water_mask)).float()

    loss_water = (error * water_mask).sum() / torch.clamp(water_mask.sum(), min=1.0)
    loss_land  = (error * land_mask).sum()  / torch.clamp(land_mask.sum(),  min=1.0)

    return water_weight * loss_water + land_weight * loss_land


# ----------------- FINAL COMBINED LOSS -----------------
def combined_loss(sr: torch.Tensor,
                  hr: torch.Tensor,
                  hr_mask: torch.Tensor,
                  water_mask: torch.Tensor = None,
                  lambda_grad: float = 0.1,
                  lambda_water: float = 0.0,
                  water_weight: float = 1.0,
                  land_weight: float = 1.0) -> torch.Tensor:

    loss = masked_l1(sr, hr, hr_mask)

    loss = loss + lambda_grad * gradient_loss(sr, hr, hr_mask)

    if water_mask is not None and lambda_water > 0.0:
        loss = loss + lambda_water * water_weighted_loss(
            sr, hr, hr_mask, water_mask,
            water_weight=water_weight,
            land_weight=land_weight,
        )

    return loss