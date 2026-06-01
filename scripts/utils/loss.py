"""
loss.py — TIR Super-Resolution Loss
Designed for thermal heterogeneity assessment in river systems (Rhône).

Structure:
    L_total = masked_l1                          (pixel accuracy: whole scene)
            + lambda_grad(t) * gradient_loss     (sharpness: whole scene, time-weighted)
            + lambda_water   * water_grad_loss   (extra sharpness: water only)

Time-aware gradient weight — two factors:

    w_contrast(lr_time):
        Solar heating curve — peaks at solar noon (~13h, summer France).
        Higher contrast at midday = gradients more meaningful = weight more.
        Range: [0.3, 1.0]

    w_gap(time_gap_hours, date_gap_days):
        Reliability of HR gradient target given acquisition gap.
        Large gap = scene changed = HR gradients less predictable from LR
        = be MORE LENIENT (reduce weight), not stricter.
        Range: [0.0, 1.0]  where 1.0 = same time (fully reliable)

    Effective weight = lambda_grad × w_contrast × w_gap

Gradient kernel — Sobel:
    3×3 weighted filter, noise-robust for TIR imagery.
    Port of prof's sobel_loss (TF) via kornia SpatialGradient.
"""

import math
import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient

_sobel = SpatialGradient(mode="sobel", normalized=True)


# =========================================================
# MASKED L1
# =========================================================
def masked_l1(sr, hr, mask):
    """L1 over valid HR pixels only (mask=1 where not NaN)."""
    err = torch.abs(sr - hr) * mask
    return err.sum() / torch.clamp(mask.sum(), min=1.0)


# =========================================================
# GRADIENT LOSS (Sobel)
# =========================================================
def gradient_loss(sr, hr, mask):
    """
    L1 on Sobel gradient tensor (x and y) over masked region.
    Mask eroded 1px to avoid boundary artifacts at valid/invalid borders.
    """
    sr_g = _sobel(sr)   # [B, 1, 2, H, W]
    hr_g = _sobel(hr)

    # Erode: only pixels where full 3×3 neighbourhood is valid
    mask_inner = (F.avg_pool2d(mask, 3, 1, 1) > 0.99).float()
    mask_g     = mask_inner.unsqueeze(2)              # [B, 1, 1, H, W]

    err = torch.abs(sr_g - hr_g) * mask_g
    return err.sum() / torch.clamp(mask_g.expand_as(err).sum(), min=1.0)


# =========================================================
# TIME-AWARE GRADIENT WEIGHT
# =========================================================

def w_contrast(lr_time, t_min=6.0, t_max=16.0,
               w_min=0.3, w_max=1.0):
    """
    Solar contrast weight based on LR acquisition time.
    Raised cosine centred at solar noon (13h, summer France).

    lr_time = 13.0 → w_max  (maximum contrast, full weight)
    lr_time =  6.0 → w_min  (low contrast morning, reduced weight)
    """
    SOLAR_NOON = 13.0
    half_width = max(SOLAR_NOON - t_min, t_max - SOLAR_NOON)
    dist       = abs(lr_time - SOLAR_NOON) / half_width
    cosine_val = 0.5 * (1.0 + math.cos(math.pi * min(dist, 1.0)))
    return w_min + (w_max - w_min) * cosine_val


def w_gap(time_gap_hours, date_gap_days,
          max_gap_hours=10.0, max_gap_days=30,
          alpha=0.7, beta=0.3):
    """
    Reliability of HR gradient target given acquisition gap.

    Large gap → scene changed → HR gradients less predictable from LR
    → REDUCE gradient weight (not increase — that would punish the model
      for something physically impossible to predict).

    Returns [0, 1]: 1.0 = same time (fully reliable), 0.0 = max gap.
    """
    norm_hour = min(abs(time_gap_hours) / max_gap_hours, 1.0)
    norm_day  = min(abs(date_gap_days)  / max_gap_days,  1.0)
    gap_score = alpha * norm_hour + beta * norm_day   # 0=no gap, 1=max gap
    return 1.0 - gap_score                            # invert: 1=reliable


def time_grad_weight(lr_time, time_gap_hours, date_gap_days, lambda_grad):
    """
    Final time-modulated gradient loss weight.
    effective_weight = lambda_grad × w_contrast(lr_time) × w_gap(gaps)

    Example (your sample patch):
        lr_time=10.37, gap=5.17h, gap=2d, lambda_grad=0.2
        w_contrast ≈ 0.80, w_gap ≈ 0.62
        effective  ≈ 0.099  (roughly half of lambda_grad)

    Example (ideal patch — same time, midday):
        lr_time=13.0, gap=0.5h, gap=0d, lambda_grad=0.2
        w_contrast = 1.00, w_gap ≈ 0.97
        effective  ≈ 0.194  (nearly full lambda_grad)
    """
    return lambda_grad * w_contrast(lr_time) * w_gap(time_gap_hours,
                                                      date_gap_days)


# =========================================================
# WATER GRADIENT LOSS
# Extra gradient pressure inside water mask only.
# Targets thermal heterogeneity in the river (scientific goal).
# No L1 component — avoids double-penalising pixel accuracy
# (already covered by masked_l1 above).
# =========================================================
def water_grad_loss(sr, hr, hr_mask, water_mask):
    """
    Gradient loss restricted to valid water pixels only.
    water pixels get:
      masked_l1    → pixel accuracy  (term 1, same as land)
      gradient     → sharpness       (term 2, same as land)
      water_grad   → EXTRA sharpness (term 3, water only)
    """
    river_mask = (hr_mask * water_mask).float()
    if river_mask.sum() < 1.0:
        return torch.tensor(0.0, device=sr.device)
    return gradient_loss(sr, hr, river_mask)


# =========================================================
# COMBINED LOSS
# =========================================================
def combined_loss(
    sr,
    hr,
    hr_mask,
    water_mask=None,
    lambda_grad=0.1,
    lambda_water=1.0,
    # Time metadata — if all provided, lambda_grad is modulated.
    # If any is None, lambda_grad used as fixed weight (safe fallback).
    lr_time=None,
    time_gap_hours=None,
    date_gap_days=None,
) -> dict:
    """
    Returns dict of loss components for wandb logging + loss_total.

    Args:
        sr             : model output      [B, 1, H, W]
        hr             : HR target         [B, 1, H, W]
        hr_mask        : valid px mask     [B, 1, H, W]
        water_mask     : water px mask     [B, 1, H, W] or None
        lambda_grad    : gradient loss weight (time-modulated if
                         lr_time/time_gap/date_gap provided)
        lambda_water   : water gradient loss weight (suggest 2.0)
        lr_time        : LR acquisition hour (scalar float, e.g. 10.37)
        time_gap_hours : abs(hr_time - lr_time) (scalar float)
        date_gap_days  : calendar day difference (scalar float)
    """
    losses = {}

    # 1. Base pixel loss — whole scene, untouched
    l_pixel = masked_l1(sr, hr, hr_mask)
    losses["l1"] = l_pixel

    # 2. Gradient loss — whole scene, time-modulated
    if (lr_time is not None
            and time_gap_hours is not None
            and date_gap_days is not None):
        w_grad = time_grad_weight(
            lr_time, time_gap_hours, date_gap_days, lambda_grad
        )
    else:
        w_grad = lambda_grad   # fixed fallback

    l_grad = gradient_loss(sr, hr, hr_mask)
    losses["loss_grad"]        = l_grad
    losses["loss_grad_weight"] = torch.tensor(
        w_grad, dtype=torch.float32, device=sr.device
    )

    # 3. Water gradient loss — water only
    l_water = torch.tensor(0.0, device=sr.device)
    if water_mask is not None and lambda_water > 0.0:
        l_water = water_grad_loss(sr, hr, hr_mask, water_mask)
    losses["loss_water"] = l_water

    # Total
    losses["loss_total"] = (
        l_pixel
        + w_grad       * l_grad
        + lambda_water * l_water
    )

    return losses