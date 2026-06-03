import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient

# Initialize Kornia Sobel filter (computes dX and dY gradients)
_sobel = SpatialGradient(mode="sobel", normalized=True)


def regional_masked_l1(sr, hr, target_mask):
    """
    Computes mean absolute error (L1) over a targeted spatial area.
    Used for both non-water (land) and water (river) absolute temperature constraints.
    """
    # Safe graph fallback: prevents breaking backpropagation if a mask is empty in a patch
    if target_mask.sum() < 1:
        return sr.sum() * 0.0
        
    err = torch.abs(sr - hr) * target_mask
    return err.sum() / torch.clamp(target_mask.sum(), min=1.0)


def gradient_loss(sr, hr, hr_mask):
    """Computes global spatial boundary error using 2D Sobel operators."""
    # sr_g and hr_g shape: [B, 1, 2, H, W] due to X and Y gradient channels
    sr_g = _sobel(sr)
    hr_g = _sobel(hr)

    # Inwardly erode the valid image mask slightly to eliminate harsh border artifacts
    mask_valid = (F.avg_pool2d(hr_mask, 3, 1, 1) > 0.99).float()
    
    # Unsqueeze at dimension 2 to match [B, 1, 1, H, W] for clean channel broadcasting
    mask_g = mask_valid.unsqueeze(2)

    err = torch.abs(sr_g - hr_g) * mask_g
    
    # Each valid mask pixel has 2 values (dX and dY), so we scale the denominator by 2.0
    return err.sum() / torch.clamp(mask_g.sum() * 2.0, min=1.0)


def time_grad_weight(time_gap_hours, 
                     date_gap_days, 
                     lambda_doy=0.3, 
                     lambda_tod=0.7):
    """
    Computes the global temporal weight vector (w_t) in range [0, 1].
    Prioritizes Time of Day (0.7) over Day of Year (0.3).
    """
    t_tod = abs(time_gap_hours) / 24.0
    t_doy = abs(date_gap_days) / 365.0
    
    t_delta = (lambda_doy * t_doy) + (lambda_tod * t_tod)
    w_t = 1.0 - t_delta
    
    return max(0.0, min(1.0, w_t))


def combined_loss(
    sr,
    hr,
    hr_mask,
    water_mask,
    lambda_nw=1.0,      # Weight multiplier for Non-Water (Land) Pixel Loss
    lambda_w=1.0,      # Weight multiplier for Water (River) Pixel Loss
    lambda_g=0.1,      # Weight multiplier for Spatial Gradient Loss (Lg)
    time_gap_hours=None,
    date_gap_days=None,
):
    """
    Main wrapper calculating the newly updated Spatiotemporally Conditioned Loss.
    Formula: L_total = w_t * ( (lambda_a * L_nw + lambda_b * L_w) + lambda_c * L_g )
    """
    
    # 1. Create mutually exclusive spatial masks for regional land and water
    # M_w = HR_mask * Water_mask
    m_w = hr_mask * water_mask
    # M_nw = HR_mask * (1 - Water_mask)
    m_nw = hr_mask * (1.0 - water_mask)

    # 2. Compute regional and structural sub-losses
    l_nw = regional_masked_l1(sr, hr, m_nw)
    l_w = regional_masked_l1(sr, hr, m_w)
    l_g = gradient_loss(sr, hr, hr_mask)

    # 3. Compute Global Temporal Weight (w_t)
    if time_gap_hours is not None and date_gap_days is not None:
        w_t = time_grad_weight(time_gap_hours, date_gap_days, 
                                lambda_doy=0.3, lambda_tod=0.7)
    else:
        w_t = 1.0

    # 4. Total Loss Execution following professor's updated bracket constraints:
    # L_total = w_t * ( (lambda_nw * L_nw + lambda_w * L_w) + lambda_g * L_g )
    thermal_bracket = (lambda_nw * l_nw) + (lambda_w * l_w)
    loss = w_t * (thermal_bracket + (lambda_g * l_g))

    return {
        "loss_total": loss,
        "loss_nw": l_nw,
        "loss_water": l_w,
        "loss_grad": l_g,
        "w_time": torch.tensor(w_t, device=sr.device),
    }