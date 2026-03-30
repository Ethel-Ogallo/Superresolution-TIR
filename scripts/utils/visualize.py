"""
visualize.py — Plotting utilities for TIR super-resolution.
Reusable across all SR models.
"""

import numpy as np
import matplotlib.pyplot as plt
import torch
from skimage.transform import resize
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


def plot_lr_hr_pair(row, nodata: float = -9999.0):
    """Sanity-check plot: show one LR / HR pair from the metadata."""
    import rasterio

    with rasterio.open(row["lr_path"]) as src:
        lr = src.read(1).astype(np.float32)
    with rasterio.open(row["hr_path"]) as src:
        hr = src.read(1).astype(np.float32)

    lr[lr == nodata] = np.nan
    hr[hr == nodata] = np.nan

    fig, axes = plt.subplots(1, 2, figsize=(10, 5))
    for ax, img, title in zip(
        axes, [lr, hr],
        [f"LR ({lr.shape[1]}×{lr.shape[0]})", f"HR ({hr.shape[1]}×{hr.shape[0]})"],
    ):
        im = ax.imshow(
            img, cmap="inferno",
            vmin=np.nanpercentile(img, 2),
            vmax=np.nanpercentile(img, 98),
        )
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Temp (°C)")

    plt.tight_layout()
    plt.show()


def visualise_predictions(
    model,
    test_loader,
    device,
    mask_sr:        bool = True,
    show_histogram: bool = True,
    model_name:     str  = "SR",
    data_range:     float = 60.0,
):
    """
    Plot LR / SR / HR side-by-side and optionally a temperature histogram.

    Args:
        mask_sr:        True  → SR masked to valid HR region (evaluation mode)
                        False → SR shown as full square (deployment demo)
        show_histogram: toggle the distribution plot
        model_name:     label shown in the SR panel title
        data_range:     temperature range for PSNR / SSIM (default 60 °C)
    """
    model.eval()
    with torch.no_grad():
        lr_t, hr_t, mask_t = next(iter(test_loader))
        lr_t, hr_t, mask_t = lr_t.to(device), hr_t.to(device), mask_t.to(device)
        sr_t = model(lr_t)

        sr_den = model.denormalize(sr_t)
        hr_den = model.denormalize(hr_t)

        # Metrics on masked valid region
        sr_crop, mask_crop = model.crop_to_valid_bbox(sr_den, mask_t)
        hr_crop, _         = model.crop_to_valid_bbox(hr_den, mask_t)
        sr_m = sr_crop * mask_crop
        hr_m = hr_crop * mask_crop

        psnr_fn  = PeakSignalNoiseRatio(data_range=data_range).to(device)
        ssim_fn  = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)
        psnr_val = psnr_fn(sr_m, hr_m).item()
        ssim_val = ssim_fn(sr_crop, hr_crop).item()

        sr  = sr_den.cpu().numpy()[0, 0]
        hr  = hr_den.cpu().numpy()[0, 0]
        lr  = model.denormalize(lr_t).cpu().numpy()[0, 0]
        mk  = mask_t.cpu().numpy()[0, 0]

    valid    = mk > 0.5
    valid_lr = resize(valid, lr.shape, order=0, preserve_range=True) > 0.5

    hr_masked  = np.where(valid, hr, np.nan)
    sr_display = np.where(valid, sr, np.nan) if mask_sr else sr

    # ── Spatial comparison ──────────────────────────────────────────────────
    sr_title = (
        f"SR ({model_name})\n"
        f"PSNR: {psnr_val:.2f} dB  |  SSIM: {ssim_val:.3f}"
    )

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    for ax, img, title in zip(
        axes,
        [lr, sr_display, hr_masked],
        ["LR Input", sr_title, "HR Target"],
    ):
        img_valid = img[~np.isnan(img)]
        vmin, vmax = np.percentile(img_valid, 2), np.percentile(img_valid, 98)
        im = ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
        ax.set_title(title)
        ax.axis("off")
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Temp (°C)")

    plt.tight_layout()
    plt.show()

    # ── Temperature distribution ─────────────────────────────────────────────
    if show_histogram:
        plt.figure(figsize=(10, 4))
        plt.hist(hr[valid].flatten(),    bins=50, alpha=0.5, label="HR", color="green")
        plt.hist(sr[valid].flatten(),    bins=50, alpha=0.5, label="SR", color="blue")
        plt.hist(lr[valid_lr].flatten(), bins=50, alpha=0.5, label="LR", color="red")
        plt.xlabel("Temperature (°C)")
        plt.ylabel("Frequency")
        plt.title("Temperature Distribution")
        plt.legend()
        plt.tight_layout()
        plt.show()