"""
scripts/utils/seq_metrics.py
"""

import torch
from torchmetrics.image import StructuralSimilarityIndexMeasure, LearnedPerceptualImagePatchSimilarity


def build_masks(hr_mask, water_mask):
    m_w = hr_mask * water_mask if water_mask is not None else None
    m_nw = hr_mask * (1.0 - water_mask) if water_mask is not None else hr_mask
    return m_nw, m_w


def _compute_metrics(sr, hr, mask, module):
    """
    Safer version that skips very small regions to prevent SSIM/LPIPS padding errors.
    """
    mask = mask > 0.5
    valid_count = mask.sum()

    if valid_count < 64:   # Minimum reasonable size
        return None, None, None, None, None

    sr_v = sr[mask]
    hr_v = hr[mask]

    mae = torch.mean(torch.abs(sr_v - hr_v))
    mse = torch.mean((sr_v - hr_v) ** 2)
    rmse = torch.sqrt(mse)
    psnr = 10 * torch.log10((module.DATA_RANGE ** 2) / (mse + 1e-8))

    # Structural & Perceptual
    ssim_scores = []
    lpips_scores = []
    
    ssim_fn = module.ssim_fn
    lpips_fn = module.lpips_fn

    for b in range(sr.shape[0]):
        valid = mask[b, 0]
        if valid.sum() < 64:
            continue

        coords = torch.where(valid)
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        h = ymax - ymin + 1
        w = xmax - xmin + 1

        if h < 16 or w < 16:
            continue

        sr_crop = sr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
        hr_crop = hr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]

        sr_norm = ((sr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)
        hr_norm = ((hr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)

        ssim_scores.append(ssim_fn(sr_norm.detach(), hr_norm.detach()))

        # LPIPS needs larger patches
        if h >= 32 and w >= 32:
            sr_3ch = sr_norm.repeat(1, 3, 1, 1).detach()
            hr_3ch = hr_norm.repeat(1, 3, 1, 1).detach()
            lpips_score = lpips_fn(sr_3ch, hr_3ch)
            lpips_scores.append(lpips_score)

    ssim = torch.stack(ssim_scores).mean() if len(ssim_scores) > 0 else torch.tensor(0.0, device=sr.device)
    lpips = torch.stack(lpips_scores).mean() if len(lpips_scores) > 0 else torch.tensor(0.0, device=sr.device)

    return psnr, ssim, mae, rmse, lpips


def compute_metrics(module, sr, hr, hr_mask, stage, water_mask=None, w_time=None):
    """Main entry point"""
    if not hasattr(module, 'ssim_fn'):
        module.ssim_fn = StructuralSimilarityIndexMeasure(data_range=1.0).to(sr.device)
    if not hasattr(module, 'lpips_fn'):
        module.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True).to(sr.device)
        module.lpips_fn.eval()

    sr = torch.nan_to_num(sr)
    hr = torch.nan_to_num(hr)

    m_nw, m_w = build_masks(hr_mask, water_mask)

    p_nw, s_nw, ma_nw, rm_nw, lp_nw = _compute_metrics(sr, hr, m_nw, module)
    
    if m_w is not None:
        p_w, s_w, ma_w, rm_w, lp_w = _compute_metrics(sr, hr, m_w, module)
    else:
        p_w = s_w = ma_w = rm_w = lp_w = None

    metrics_dict = {
        "land_psnr": p_nw, 
        "land_ssim": s_nw, 
        "land_mae": ma_nw, 
        "land_rmse": rm_nw,
        "land_lpips": lp_nw,
        "water_psnr": p_w, 
        "water_ssim": s_w, 
        "water_mae": ma_w, 
        "water_rmse": rm_w,
        "water_lpips": lp_w,
    }

    if w_time is not None:
        metrics_dict["w_time"] = w_time.detach() if isinstance(w_time, torch.Tensor) else torch.tensor(w_time, device=sr.device)

    return metrics_dict