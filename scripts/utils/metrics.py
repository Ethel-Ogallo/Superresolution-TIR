"""
scripts/utils/metrics.py — Metrics for spatial decoupling of TIR data
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure

def build_masks(hr_mask, water_mask):
    """
    Decouples the spatial field into water and non-water components.
    """
    m_w = hr_mask * water_mask if water_mask is not None else None
    m_nw = hr_mask * (1.0 - water_mask) if water_mask is not None else hr_mask
    return m_nw, m_w

def _compute_metrics(sr, hr, mask, module):
    """
    Internal helper to compute physical metrics (MAE, RMSE, PSNR, SSIM).
    Returns None if insufficient data exists to prevent biased averaging.
    """
    mask = mask > 0.5
    
    # Return None if the mask is empty or too small for statistical significance
    if mask.sum() < 11:
        return None, None, None, None

    sr_v = sr[mask]
    hr_v = hr[mask]

    # Metrics calculated in actual physical units (°C)
    mae = torch.mean(torch.abs(sr_v - hr_v))
    mse = torch.mean((sr_v - hr_v) ** 2)
    rmse = torch.sqrt(mse)
    
    # PSNR based on the physical range of the dataset
    psnr = 10 * torch.log10((module.DATA_RANGE ** 2) / (mse + 1e-8))

    # Structural Similarity (SSIM)
    ssim_scores = []
    ssim_fn = module.ssim_fn  # Use the module's initialized SSIM object for efficiency

    for b in range(sr.shape[0]):
        valid = mask[b, 0]
        if valid.sum() < 11: continue

        coords = torch.where(valid)
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        if (ymax - ymin < 2) or (xmax - xmin < 2): continue

        sr_crop = sr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
        hr_crop = hr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]

        sr_norm = ((sr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)
        hr_norm = ((hr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)

        ssim_scores.append(ssim_fn(sr_norm.detach(), hr_norm.detach()))

    ssim = torch.stack(ssim_scores).mean() if len(ssim_scores) > 0 else torch.tensor(0.0, device=sr.device)

    return psnr, ssim, mae, rmse

def compute_metrics(module, sr, hr, hr_mask, stage, water_mask=None, w_time=None):
    """
    Main entry point for calculating land and river metrics.
    """
    # Initialize SSIM object once and attach to module for efficiency
    if not hasattr(module, 'ssim_fn'):
        module.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(sr.device)

    sr = torch.nan_to_num(sr)
    hr = torch.nan_to_num(hr)

    m_nw, m_w = build_masks(hr_mask, water_mask)

    # Land Metrics
    p_nw, s_nw, ma_nw, rm_nw = _compute_metrics(sr, hr, m_nw, module)
    
    # River Metrics (Only if water_mask exists)
    p_w, s_w, ma_w, rm_w = _compute_metrics(sr, hr, m_w, module) if m_w is not None else (None,)*4

    metrics_dict = {
        "land_psnr": p_nw, 
        "land_ssim": s_nw, 
        "land_mae": ma_nw, 
        "land_rmse": rm_nw,
        "water_psnr": p_w, 
        "water_ssim": s_w, 
        "water_mae": ma_w, 
        "water_rmse": rm_w,
    }

    if w_time is not None:
        metrics_dict["w_time"] = w_time.detach() if isinstance(w_time, torch.Tensor) else torch.tensor(w_time, device=sr.device)

    return metrics_dict