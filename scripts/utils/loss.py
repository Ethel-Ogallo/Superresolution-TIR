"""
loss.py — TIR Super-Resolution Loss
Designed for thermal heterogeneity assessment in water systems (Rhône).

Structure:
    L_total = masked_l1                         (pixel accuracy: whole scene)
            + lambda_grad  * gradient_loss      (sharpness: whole scene)
            + lambda_water * water_grad_loss    (extra sharpness: water only)

Why this structure:
    - masked_l1 covers absolute temperature accuracy everywhere equally
    - gradient_loss encourages sharp edges across the whole scene
    - water_grad_loss adds extra gradient pressure on the water because
      thermal spatial heterogeneity (the scientific goal) is measured
      by within-water temperature gradients, not just pixel accuracy.
      water pixels already get L1 + gradient from terms 1 and 2 —
      term 3 adds focused gradient pressure without double-penalising
      pixel accuracy.

Suggested starting weights:
    lambda_grad  = 0.1   gradient loss over whole scene
    lambda_water = 2.0   extra gradient pressure on water
                         (high because water is primary target)
"""

import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient


# Gradient estimation via Sobel kernel — 3×3 weighted filter that
# estimates spatial gradients while suppressing sensor noise.
# TIR sensors have inherent detector noise (weak emitted signal →
# sensitive detector → electronic noise). Simple finite differences
# (pixel[i+1] - pixel[i]) amplify this noise. Sobel averages over
# neighbours before differencing, suppressing noise amplification.
# Equivalent to tf.image.sobel_edges from prof's sobel_loss in TF.
_sobel = SpatialGradient(mode="sobel", normalized=True)


# =========================================================
# MASKED L1
# Base loss — pixel accuracy over all valid HR pixels.
# Treats every pixel equally. Not modified for water/land.
# =========================================================
def masked_l1(sr, hr, mask):
    """
    L1 loss restricted to valid HR pixels (mask=1 where HR is not NaN).

    Args:
        sr   : SR output  [B, 1, H, W]
        hr   : HR target  [B, 1, H, W]
        mask : hr_mask    [B, 1, H, W]  1=valid, 0=invalid
    """
    err = torch.abs(sr - hr) * mask
    return err.sum() / torch.clamp(mask.sum(), min=1.0)


# =========================================================
# GRADIENT LOSS
# Sharpness loss — penalises blurry edges.
# Port of prof's sobel_loss (TF) to PyTorch via kornia.
# =========================================================
def gradient_loss(sr, hr, mask):
    """
    L1 on Sobel gradient tensor (x and y directions) over masked region.

    Mask is eroded by 1px to avoid boundary artifacts where the 3×3
    Sobel kernel overlaps valid/invalid pixel borders — without erosion,
    gradients at mask boundaries are computed from partially invalid
    neighbourhoods and produce spurious large values.

    Args:
        sr   : SR output  [B, 1, H, W]
        hr   : HR target  [B, 1, H, W]
        mask : any binary mask [B, 1, H, W]  (hr_mask or water_mask)
    """
    sr_g = _sobel(sr)   # [B, 1, 2, H, W]  — 2: (dx, dy)
    hr_g = _sobel(hr)   # [B, 1, 2, H, W]

    # Erode: only pixels where full 3×3 neighbourhood is valid
    mask_inner = (F.avg_pool2d(mask, 3, 1, 1) > 0.99).float()

    # Unsqueeze to broadcast over both gradient directions (dim=2)
    mask_g = mask_inner.unsqueeze(2)                        # [B, 1, 1, H, W]
    err    = torch.abs(sr_g - hr_g) * mask_g

    return err.sum() / torch.clamp(mask_g.expand_as(err).sum(), min=1.0)


# =========================================================
# water GRADIENT LOSS
# Extra gradient pressure computed ONLY inside the water mask.
# Directly targets thermal heterogeneity preservation in the
# water — the core scientific goal of this project.
#
# water pixels receive:
#   masked_l1     → pixel accuracy   (term 1, same as land)
#   gradient_loss → edge sharpness   (term 2, same as land)
#   water_grad_loss → EXTRA sharpness (term 3, water only)
#
# Term 3 adds no new L1 penalty (no double-counting pixel accuracy).
# It only adds extra gradient pressure where it scientifically matters.
# =========================================================
def water_grad_loss(sr, hr, hr_mask, water_mask):
    """
    Gradient loss restricted to valid water pixels only.

    Args:
        sr         : SR output    [B, 1, H, W]
        hr         : HR target    [B, 1, H, W]
        hr_mask    : valid pixels [B, 1, H, W]
        water_mask : water pixels [B, 1, H, W]
    """
    # Intersection: valid HR pixels that are also water
    water_mask = (hr_mask * water_mask).float()

    # Skip gracefully if no water pixels in this batch
    # (can happen for patches that are all land)
    if water_mask.sum() < 1.0:
        return torch.tensor(0.0, device=sr.device)

    return gradient_loss(sr, hr, water_mask)


# =========================================================
# COMBINED LOSS
# =========================================================
def combined_loss(
    sr,
    hr,
    hr_mask,
    water_mask=None,
    lambda_grad=0.1,
    lambda_water=0.0,
) -> dict:
    """
    Args:
        sr           : model output      [B, 1, H, W]
        hr           : HR target         [B, 1, H, W]
        hr_mask      : valid px mask     [B, 1, H, W]
        water_mask   : water px mask     [B, 1, H, W] or None
        lambda_grad  : gradient loss weight (whole scene)
        lambda_water : water gradient loss weight
                       set to 0.0 to disable (e.g. land-only patches)

    Returns:
        dict of individual loss components + loss_total.
        Returning a dict lets shared_step log each to wandb
        without extra boilerplate.y
    """
    losses = {}

    # 1. Base pixel loss — whole scene, untouched
    l1 = masked_l1(sr, hr, hr_mask)
    losses["l1"] = l1

    # 2. Gradient loss — whole scene
    l_grad = gradient_loss(sr, hr, hr_mask)
    losses["loss_grad"] = l_grad

    # 3. water gradient loss — water only
    # Zero if water_mask not provided or lambda_water=0
    l_water = torch.tensor(0.0, device=sr.device)
    if water_mask is not None and lambda_water > 0.0:
        l_water = water_grad_loss(sr, hr, hr_mask, water_mask)
    losses["loss_water"] = l_water

    # Total
    losses["loss_total"] = (
        l1
        + lambda_grad  * l_grad
        + lambda_water * l_water
    )

    return losses