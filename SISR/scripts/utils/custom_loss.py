import torch
import torch.nn.functional as F

def regional_masked_l1_batch(sr, hr, mask):
    """
    Fully vectorized masked L1 over [B, 1, H, W] tensors.
    Each sample normalized by its own mask pixel count.
    """
    err        = torch.abs(sr - hr) * mask                        # [B, 1, H, W]
    per_sample = err.sum(dim=(1, 2, 3))                           # [B]
    counts     = mask.sum(dim=(1, 2, 3)).clamp(min=1.0)           # [B]
    return per_sample / counts                                     # [B]  — per-sample loss


# def gradient_loss_batch(sr, hr, hr_mask):
#     """
#     Vectorized gradient magnitude MAE over [B, 1, H, W].
#     Returns per-sample scalar tensor [B].
#     """
#     sr_dx = sr[:, :, :, 1:]  - sr[:, :, :, :-1]
#     sr_dy = sr[:, :, 1:, :]  - sr[:, :, :-1, :]
#     hr_dx = hr[:, :, :, 1:]  - hr[:, :, :, :-1]
#     hr_dy = hr[:, :, 1:, :]  - hr[:, :, :-1, :]

#     mask_valid = (F.avg_pool2d(hr_mask, 3, 1, 1) > 0.99).float()
#     mask_dx    = mask_valid[:, :, :, :-1]
#     mask_dy    = mask_valid[:, :, :-1, :]

#     sr_mag = torch.sqrt(
#         (sr_dx * mask_dx)[:, :, :-1, :] ** 2 +
#         (sr_dy * mask_dy)[:, :, :, :-1] ** 2 + 1e-8
#     )
#     hr_mag = torch.sqrt(
#         (hr_dx * mask_dx)[:, :, :-1, :] ** 2 +
#         (hr_dy * mask_dy)[:, :, :, :-1] ** 2 + 1e-8
#     )

#     loss_map = torch.abs(sr_mag - hr_mag)                         # [B, 1, 255, 255]
#     return loss_map.mean(dim=(1, 2, 3))                           # [B]

def gradient_loss_batch(sr, hr, hr_mask):
    """
    Masked gradient-magnitude MAE over [B, 1, H, W].
    Returns one loss per sample: [B].
    """

    # Forward horizontal and vertical gradients.
    sr_dx = sr[:, :, :-1, 1:] - sr[:, :, :-1, :-1]
    sr_dy = sr[:, :, 1:, :-1] - sr[:, :, :-1, :-1]

    hr_dx = hr[:, :, :-1, 1:] - hr[:, :, :-1, :-1]
    hr_dy = hr[:, :, 1:, :-1] - hr[:, :, :-1, :-1]

    # Gradient magnitudes.
    sr_mag = torch.sqrt(sr_dx.square() + sr_dy.square() + 1e-8)
    hr_mag = torch.sqrt(hr_dx.square() + hr_dy.square() + 1e-8)

    # A gradient cell is valid only if all four surrounding pixels are valid.
    mask = (
        hr_mask[:, :, :-1, :-1]
        * hr_mask[:, :, :-1, 1:]
        * hr_mask[:, :, 1:, :-1]
        * hr_mask[:, :, 1:, 1:]
    )

    return regional_masked_l1_batch(sr_mag, hr_mag, mask)

def compute_wt_batch(time_channels, lambda_doy=0.3, lambda_tod=0.7, clamp_min=0.2):
    """
    Calculate temporal weights from normalized time channels.
    time_channels: [B, 2, H, W]
    Returns: [B]
    """

    t_tod = time_channels[:, 0].abs().mean(dim=(1, 2))
    t_doy = time_channels[:, 1].abs().mean(dim=(1, 2))

    t_delta = lambda_doy * t_doy + lambda_tod * t_tod
    w_t = (1.0 - t_delta).clamp(min=clamp_min, max=1.0)

    return w_t  # [B]

def combined_loss(sr, hr, hr_mask, water_mask,
                  lambda_nw, lambda_w, lambda_g,
                  time_channels=None):
    """
    Fully vectorized combined loss. No Python loop over batch.
    All per-sample operations run in parallel on GPU.
    """
    # Masks 
    m_w  = hr_mask * water_mask                                   
    m_nw = hr_mask * (1.0 - water_mask)                         

    # Per-sample sub-losses [B] 
    l_nw = regional_masked_l1_batch(sr, hr, m_nw)                
    l_w  = regional_masked_l1_batch(sr, hr, m_w)                
    l_g  = gradient_loss_batch(sr, hr, hr_mask)               

    # Scale normalization so lambdas are directly comparable
    eps   = 1e-2
    l_nw_n = l_nw / (l_nw.detach().mean() + eps)
    l_w_n  = l_w  / (l_w.detach().mean()  + eps)
    l_g_n  = l_g  / (l_g.detach().mean()  + eps)

    # Temporal weights [B]
    if time_channels is not None:
        w_t = compute_wt_batch(time_channels.to(sr.device))
    else:
        w_t = torch.ones(sr.shape[0], device=sr.device, dtype=sr.dtype,)

    # Weighted custom loss.
    per_sample = w_t * (lambda_nw * l_nw + lambda_w * l_w + lambda_g * l_g)

    return {
        "loss_total": per_sample.mean(),
        "loss_nw": l_nw.mean(),
        "loss_water": l_w.mean(),
        "loss_grad": l_g.mean(),
        "w_time": w_t.mean(),
    }