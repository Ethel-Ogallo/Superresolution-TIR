import torch
import torch.nn.functional as F
from kornia.filters import SpatialGradient

# Initialize Kornia Sobel filter (computes horizontal dX and vertical dY gradients)
_sobel = SpatialGradient(mode="sobel", normalized=True)


def regional_masked_l1(sr, hr, target_mask):
    """
    Computes Mean Absolute Error (MAE) exclusively over a specific spatial mask.
    This function isolates either the river water or the land surface.
    """
    # Safe graph fallback: If a patch has no valid mask pixels, prevent dividing by zero
    if target_mask.sum() < 1:
        return sr.sum() * 0.0
        
    err = torch.abs(sr - hr) * target_mask
    return err.sum() / torch.clamp(target_mask.sum(), min=1.0)


# def gradient_loss(sr, hr, hr_mask):
#     """
#     Computes global structural edge error using 2D Sobel operators.
#     Measures how sharp and aligned the riverbanks and geometric boundaries are.
#     """
#     # Outputs shape: [B, 1, 2, H, W] representing structural gradient components
#     sr_g = _sobel(sr)
#     hr_g = _sobel(hr)

#     # Inwardly erode the image border slightly to eliminate harsh edge clipping artifacts
#     mask_valid = (F.avg_pool2d(hr_mask, 3, 1, 1) > 0.99).float()
#     mask_g = mask_valid.unsqueeze(2) # Broadened to channel broadcast smoothly

#     err = torch.abs(sr_g - hr_g) * mask_g
    
#     # Scale denominator by 2.0 because each pixel has a horizontal and vertical derivative
#     return err.sum() / torch.clamp(mask_g.sum() * 2.0, min=1.0)
def gradient_loss(sr, hr, hr_mask):
    """
    PyTorch translation of tf.image.image_gradients() loss.
    Computes the magnitude of pure pixel-to-pixel spatial gradients.
    """
    # 1. Compute horizontal and vertical pixel-to-pixel differences
    # sr_dx shape: [B, C, 256, 255] | sr_dy shape: [B, C, 255, 256]
    sr_dx = sr[:, :, :, 1:] - sr[:, :, :, :-1]
    sr_dy = sr[:, :, 1:, :] - sr[:, :, :-1, :]

    hr_dx = hr[:, :, :, 1:] - hr[:, :, :, :-1]
    hr_dy = hr[:, :, 1:, :] - hr[:, :, :-1, :]

    # 2. Match the valid mask to the reduced dimensions from diffing
    mask_valid = (F.avg_pool2d(hr_mask, 3, 1, 1) > 0.99).float()
    mask_dx = mask_valid[:, :, :, :-1]
    mask_dy = mask_valid[:, :, :-1, :]

    # 3. Mask them independently first
    sr_norm_x = sr_dx * mask_dx  # Shape: [B, C, 256, 255]
    sr_norm_y = sr_dy * mask_dy  # Shape: [B, C, 255, 256]
    hr_norm_x = hr_dx * mask_dx  # Shape: [B, C, 256, 255]
    hr_norm_y = hr_dy * mask_dy  # Shape: [B, C, 255, 256]

    # --- THE EDIT: Slice both arrays down to a shared 255x255 footprint ---
    # Slice norm_x along height (dim 2) -> [B, C, 255, 255]
    # Slice norm_y along width (dim 3)  -> [B, C, 255, 255]
    sr_magnitude = torch.sqrt(
        sr_norm_x[:, :, :-1, :] ** 2 + sr_norm_y[:, :, :, :-1] ** 2 + 1e-8
    )
    hr_magnitude = torch.sqrt(
        hr_norm_x[:, :, :-1, :] ** 2 + hr_norm_y[:, :, :, :-1] ** 2 + 1e-8
    )

    # 4. Return the Mean Absolute Error between the two gradient fields
    # Both magnitude tensors are now perfectly matched at [B, C, 255, 255]
    loss = torch.abs(sr_magnitude - hr_magnitude)

    return loss.mean()

def time_grad_weight(time_gap_hours, 
                     date_gap_days, 
                     lambda_doy=0.3, 
                     lambda_tod=0.7):
    """
    Computes the Global Temporal Weight (w_t) based on acquisition time mismatches.
    Penalizes image pairs captured at different hours of the day or seasons of the year.
    Prioritizes Time of Day (70% weight) over Day of Year (30% weight) due to diurnal cycle shifts.
    """
    t_tod = abs(time_gap_hours) / 24.0  # Fraction of day mismatch
    t_doy = abs(date_gap_days) / 365.0   # Fraction of year mismatch
    
    t_delta = (lambda_doy * t_doy) + (lambda_tod * t_tod)
    w_t = 1.0 - t_delta                 # Converts error into a retention score [0, 1]
    
    return max(0.0, min(1.0, w_t))


def combined_loss(
    sr,
    hr,
    hr_mask,
    water_mask,
    lambda_nw=1.0,      # Hyperparameter: Multiplier for Land Terrain
    lambda_w=1.0,       # Hyperparameter: Multiplier for River Water Channel
    lambda_g=0.1,       # Hyperparameter: Multiplier for Sobel Gradient Boundaries
    time_gap_hours=None,
    date_gap_days=None,
):
    """
    Main Optimization Objective Engine.
    Formula: Total_Loss = Global_Time_Weight * ( (Land_Weight * Land_L1) + (Water_Weight * Water_L1) + (Sobel_Weight * Structural_Edge_L1) )
    """
    
    # 1. Create mutually exclusive spatial domain masks
    m_w = hr_mask * water_mask              # Matrix: Only Water Channel
    m_nw = hr_mask * (1.0 - water_mask)     # Matrix: Only Surrounding Land Terrain

    # 2. Compute raw sub-losses (Expressed in true physical °C scales)
    l_nw = regional_masked_l1(sr, hr, m_nw) # RAW_LAND_LOSS
    l_w = regional_masked_l1(sr, hr, m_w)   # RAW_WATER_LOSS
    l_g = gradient_loss(sr, hr, hr_mask)    # RAW_STRUCTURAL_EDGE_LOSS

    # 3. Compute Global Temporal Scaling Factor (w_t)
    if time_gap_hours is not None and date_gap_days is not None:
        w_t = time_grad_weight(time_gap_hours, date_gap_days, 
                                lambda_doy=0.3, lambda_tod=0.7)
    else:
        w_t = 1.0

    # 4. Total Loss Compilation following your equation brackets:
    # L_total = w_t * ( (lambda_nw * L_nw + lambda_w * L_w) + lambda_g * L_g )
    thermal_bracket = (lambda_nw * l_nw) + (lambda_w * l_w)
    loss = w_t * (thermal_bracket + (lambda_g * l_g))

    # Return dictionary feeding directly into our newly updated logging hooks
    return {
        "loss_total": loss,         # BUNDLED RECON LOSS FOR OPTIMIZER
        "loss_nw": l_nw,           # Raw unweighted land absolute error
        "loss_water": l_w,         # Raw unweighted river absolute error
        "loss_grad": l_g,          # Raw unweighted boundary gradient error
        "w_time": torch.tensor(w_t, device=sr.device), # Raw calculated temporal scalar
    }