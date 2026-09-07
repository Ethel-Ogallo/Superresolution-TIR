"""
compare_models.py — Load all best checkpoints and produce a unified
benchmark table + side-by-side visual comparison.

Expects a simple config file (models.csv) listing model name and checkpoint
path for each run you want to compare. Example models.csv:

    name,checkpoint
    EDSR,wandb:entity/project/model-abc123:best
    SwinIR,wandb:entity/project/model-def456:best
    HAT,checkpoints/hat_best.ckpt

Output:
    results/figures/comparison_table.csv     ← PSNR / SSIM per model
    results/figures/comparison_grid.png      ← visual side-by-side grid
    results/figures/comparison_boxplot.png   ← PSNR distribution per model

Launch via:  sbatch jobs/eval_all.slurm
"""

import argparse
import os

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.data_prep import SRDataset
from scripts.evaluation.evaluate import load_model


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def parse_args():
    p = argparse.ArgumentParser(
        description="Benchmark all SR models on the test set"
    )
    p.add_argument("--models_csv",   default="models.csv",
                   help="CSV with columns: name, checkpoint")
    p.add_argument("--metadata_csv", default="metadata.csv")
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--n_vis",        type=int, default=3,
                   help="Number of patches used in the side-by-side visual grid")
    p.add_argument("--data_range",   type=float, default=60.0)
    return p.parse_args()


# ---------------------------------------------------------------------------
# Per-patch metrics
# ---------------------------------------------------------------------------

@torch.no_grad()
def compute_patch_metrics(
    model, loader, device, data_range: float
) -> list[dict]:
    """Return per-patch PSNR and SSIM for a model."""
    psnr_fn = PeakSignalNoiseRatio(data_range=data_range).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=data_range).to(device)

    records = []
    for lr_t, hr_t, mask_t in loader:
        lr_t, hr_t, mask_t = lr_t.to(device), hr_t.to(device), mask_t.to(device)
        sr_t = model(lr_t)

        sr = model.denormalize(sr_t)
        hr = model.denormalize(hr_t)

        sr_crop, mask_crop = model.crop_to_valid_bbox(sr, mask_t)
        hr_crop, _         = model.crop_to_valid_bbox(hr, mask_t)

        psnr = psnr_fn(sr_crop * mask_crop, hr_crop * mask_crop).item()
        ssim = ssim_fn(sr_crop, hr_crop).item()
        records.append({"psnr": psnr, "ssim": ssim})

    return records


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_comparison_table(summary: pd.DataFrame, out_path: str):
    """Bar chart of mean PSNR and SSIM per model."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    for ax, metric, ylabel in zip(
        axes,
        ["psnr_mean", "ssim_mean"],
        ["PSNR (dB)", "SSIM"],
    ):
        bars = ax.bar(summary["name"], summary[metric],
                      yerr=summary[metric.replace("mean", "std")],
                      capsize=4, color="steelblue", edgecolor="white")
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel)
        ax.set_xticks(range(len(summary)))
        ax.set_xticklabels(summary["name"], rotation=15, ha="right")
        for bar, val in zip(bars, summary[metric]):
            ax.text(bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.002,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=8)

    plt.suptitle("Model Benchmark — Test Set", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def plot_boxplot(all_records: dict[str, list[dict]], out_path: str):
    """Box plot of per-patch PSNR distribution per model."""
    fig, ax = plt.subplots(figsize=(10, 5))
    names  = list(all_records.keys())
    data   = [[r["psnr"] for r in all_records[n]] for n in names]

    bp = ax.boxplot(data, patch_artist=True, notch=False)
    for patch in bp["boxes"]:
        patch.set_facecolor("steelblue")
        patch.set_alpha(0.7)

    ax.set_xticks(range(1, len(names) + 1))
    ax.set_xticklabels(names, rotation=15, ha="right")
    ax.set_ylabel("PSNR (dB)")
    ax.set_title("PSNR Distribution per Model — Test Set")
    ax.grid(axis="y", linestyle="--", alpha=0.5)

    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


def plot_visual_grid(
    models_dict: dict,   # {name: model}
    ds: SRDataset,
    device: torch.device,
    patch_indices: list[int],
    out_path: str,
):
    """
    Side-by-side grid: rows = patches, cols = LR | model_1 | model_2 | … | HR
    """
    n_patches = len(patch_indices)
    n_models  = len(models_dict)
    n_cols    = 2 + n_models   # LR + models + HR

    fig, axes = plt.subplots(
        n_patches, n_cols,
        figsize=(4 * n_cols, 4 * n_patches),
        squeeze=False,
    )

    col_titles = ["LR"] + list(models_dict.keys()) + ["HR"]
    for ax, title in zip(axes[0], col_titles):
        ax.set_title(title, fontsize=11, fontweight="bold")

    for row, idx in enumerate(patch_indices):
        lr_t, hr_t, mask_t = ds[idx]
        lr_t   = lr_t.unsqueeze(0).to(device)
        hr_t   = hr_t.unsqueeze(0).to(device)
        mask_t = mask_t.unsqueeze(0).to(device)

        # Use first model to denormalize (all share same mean/std from metadata)
        first_model = next(iter(models_dict.values()))
        lr_den = first_model.denormalize(lr_t).cpu().numpy()[0, 0]
        hr_den = first_model.denormalize(hr_t).cpu().numpy()[0, 0]
        mk     = mask_t.cpu().numpy()[0, 0] > 0.5

        images = [lr_den]
        for model in models_dict.values():
            with torch.no_grad():
                sr_t = model(lr_t)
            sr_den = model.denormalize(sr_t).cpu().numpy()[0, 0]
            images.append(np.where(mk, sr_den, np.nan))
        images.append(np.where(mk, hr_den, np.nan))

        # shared colour scale from HR valid pixels
        hr_valid = hr_den[mk]
        vmin, vmax = np.percentile(hr_valid, 2), np.percentile(hr_valid, 98)

        for ax, img in zip(axes[row], images):
            im = ax.imshow(img, cmap="inferno", vmin=vmin, vmax=vmax)
            ax.axis("off")
            plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    plt.suptitle("Visual Comparison — Test Patches", fontsize=13, fontweight="bold")
    plt.tight_layout()
    plt.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  → {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    os.makedirs("results/figures", exist_ok=True)

    # ── Load model list ──────────────────────────────────────────────────────
    if not os.path.exists(args.models_csv):
        raise FileNotFoundError(
            f"{args.models_csv} not found.\n"
            "Create it with columns: name, checkpoint\n"
            "Example row: EDSR, wandb:entity/project/model-abc:best"
        )

    model_list = pd.read_csv(args.models_csv)
    print(f"Comparing {len(model_list)} models on '{args.split}' split\n")

    # ── Data ────────────────────────────────────────────────────────────────
    metadata = pd.read_csv(args.metadata_csv)
    subset   = metadata[metadata["split"] == args.split].reset_index(drop=True)

    # ── Load all models ──────────────────────────────────────────────────────
    models_dict  = {}   # {display_name: model}
    all_records  = {}   # {display_name: [{"psnr":…, "ssim":…}, …]}
    summary_rows = []

    for _, row in model_list.iterrows():
        name = row["name"]
        print(f"Loading {name} …")
        model, _ = load_model(row["checkpoint"], device)

        ds     = SRDataset(subset, do_augment=False,
                           mean=model.hparams.mean, std=model.hparams.std)
        loader = DataLoader(ds, batch_size=1, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)

        records = compute_patch_metrics(model, loader, device, args.data_range)
        psnr_vals = [r["psnr"] for r in records]
        ssim_vals = [r["ssim"] for r in records]

        models_dict[name] = model
        all_records[name] = records

        summary_rows.append({
            "name":      name,
            "psnr_mean": np.mean(psnr_vals),
            "psnr_std":  np.std(psnr_vals),
            "ssim_mean": np.mean(ssim_vals),
            "ssim_std":  np.std(ssim_vals),
        })
        print(f"  PSNR {np.mean(psnr_vals):.2f} ± {np.std(psnr_vals):.2f} dB  |  "
              f"SSIM {np.mean(ssim_vals):.3f} ± {np.std(ssim_vals):.3f}\n")

    # ── Summary table ────────────────────────────────────────────────────────
    summary = pd.DataFrame(summary_rows).sort_values("psnr_mean", ascending=False)
    table_path = "results/figures/comparison_table.csv"
    summary.to_csv(table_path, index=False)
    print(f"  → {table_path}")
    print(summary.to_string(index=False))

    # ── Plots ────────────────────────────────────────────────────────────────
    print("\nGenerating plots …")
    plot_comparison_table(summary, "results/figures/comparison_bars.png")
    plot_boxplot(all_records,      "results/figures/comparison_boxplot.png")

    # Visual grid — use first ds (mean/std same for all models sharing metadata)
    ds_vis = SRDataset(
        subset, do_augment=False,
        mean=next(iter(models_dict.values())).hparams.mean,
        std=next(iter(models_dict.values())).hparams.std,
    )
    patch_indices = list(range(min(args.n_vis, len(ds_vis))))
    plot_visual_grid(
        models_dict, ds_vis, device, patch_indices,
        "results/figures/comparison_grid.png",
    )

    print("\nDone.")


if __name__ == "__main__":
    main()