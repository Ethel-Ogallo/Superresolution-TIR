import torch
import torch.nn.functional as F
from torchmetrics.image import StructuralSimilarityIndexMeasure


def _get_ssim(device):
    return StructuralSimilarityIndexMeasure(data_range=1.0).to(device)


def build_masks(hr_mask, water_mask):

    # M_w = HR_mask * Water_mask
    m_w = hr_mask * water_mask if water_mask is not None else None
    
    # M_nw = HR_mask * (1 - Water_mask)
    m_nw = hr_mask * (1.0 - water_mask) if water_mask is not None else hr_mask
    
    return m_nw, m_w


def _compute_metrics(sr, hr, mask, module):
    mask = mask > 0.5

    if mask.sum() == 0:
        z = torch.tensor(0.0, device=sr.device)
        return z, z, z, z

    sr_v = sr[mask]
    hr_v = hr[mask]

    # MAE and RMSE are calculated in true degrees Celsius (°C)
    mae = torch.mean(torch.abs(sr_v - hr_v))
    mse = torch.mean((sr_v - hr_v) ** 2)
    rmse = torch.sqrt(mse)

    psnr = 10 * torch.log10((module.DATA_RANGE ** 2) / (mse + 1e-8))

    ssim_fn = _get_ssim(sr.device)
    ssim_scores = []

    for b in range(sr.shape[0]):
        valid = mask[b, 0] if mask.dim() == 4 else mask[b]

        # SSIM requires a window size minimum (typically 11x11 patch size)
        if valid.sum() < 11:
            continue

        coords = torch.where(valid)
        ymin, ymax = coords[0].min(), coords[0].max()
        xmin, xmax = coords[1].min(), coords[1].max()

        # Prevent out-of-bounds tight crops from throwing exceptions
        if (ymax - ymin < 2) or (xmax - xmin < 2):
            continue

        sr_crop = sr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]
        hr_crop = hr[b:b+1, :, ymin:ymax+1, xmin:xmax+1]

        sr_norm = ((sr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)
        hr_norm = ((hr_crop - module.DATA_MIN) / module.DATA_RANGE).clamp(0, 1)

        ssim_scores.append(ssim_fn(sr_norm.detach(), hr_norm.detach()))

    ssim = torch.stack(ssim_scores).mean() if len(ssim_scores) > 0 else torch.tensor(0.0, device=sr.device)

    return psnr, ssim, mae, rmse


def compute_metrics(module, sr, hr, hr_mask, stage, water_mask=None, w_time=None):

    sr = torch.nan_to_num(sr)
    hr = torch.nan_to_num(hr)

    # 1. Build decoupled spatial masks
    m_nw, m_w = build_masks(hr_mask, water_mask)

    # 2. Compute separate metrics for Land (Non-Water) regions
    p_nw, s_nw, ma_nw, rm_nw = _compute_metrics(sr, hr, m_nw, module)

    # 3. Compute separate metrics for River Channel (Water) regions
    if m_w is not None:
        p_w, s_w, ma_w, rm_w = _compute_metrics(sr, hr, m_w, module)
    else:
        p_w = s_w = ma_w = rm_w = None

    # 4. Construct clean logging tracking dictionary for W&B
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

    # 5. Log the clean un-normalized time weight to monitor diurnal tracking
    if w_time is not None:
        # Check if tensor already to avoid device conversion latency
        if isinstance(w_time, torch.Tensor):
            metrics_dict["w_time"] = w_time.detach()
        else:
            metrics_dict["w_time"] = torch.tensor(w_time, device=sr.device)

    return metrics_dict