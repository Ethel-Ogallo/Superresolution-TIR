"""
evaluate.py — Full evaluation pipeline for TIR SR models.

Two evaluation levels:
  1. Patch-level  : PSNR, SSIM, MAE per patch
  2. Scene-level  : reassemble all patches per hr_image_id into full image,
                    compute global MAE vs full HR ground truth

Metadata is a JSON file (produced by create_metadata.py) with fields:
    hr_image_id, patch_id, hr_path, lr_path, hr_row, hr_col,
    hr_transform, crs, split

For scene-level evaluation, pass --hr_full_dir pointing to the folder
containing the original full HR GeoTIFFs, named {hr_image_id}.tif

Launch via: sbatch jobs/eval_edsr.slurm
"""

import argparse
import os

import json
import numpy as np
import pandas as pd
import rasterio
from rasterio.transform import Affine
import torch
import wandb
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.dataset import SRDataset


# -------------------- Model loading -------------------------------
def load_model(ckpt_path: str, device: torch.device):
    from scripts.models.edsr import EDSRModule
    # from scripts.models.swinir import SwinIRModule
    # from scripts.models.hat    import HATModule
    REGISTRY = {"EDSRModule": EDSRModule}

    if ckpt_path.startswith("wandb:"):
        api   = wandb.Api()
        art   = api.artifact(ckpt_path.removeprefix("wandb:"))
        local = art.download(root="/tmp/ckpt")
        ckpt_path = next(
            os.path.join(local, f) for f in os.listdir(local) if f.endswith(".ckpt")
        )
        print(f"[INFO] Downloaded checkpoint → {ckpt_path}")

    raw      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls_name = raw.get("hyper_parameters", {}).get("model_class", None)

    if cls_name and cls_name in REGISTRY:
        model = REGISTRY[cls_name].load_from_checkpoint(ckpt_path, map_location=device)
        print(f"[INFO] Loaded {cls_name}")
        return model.eval().to(device)

    for name, cls in REGISTRY.items():
        try:
            model = cls.load_from_checkpoint(ckpt_path, map_location=device)
            print(f"[INFO] Loaded {name} (fallback)")
            return model.eval().to(device)
        except Exception:
            continue

    raise ValueError("Could not load model. Register the class in evaluate.py::REGISTRY.")


# ----------------- Patch-level metrics -------------------------------
@torch.no_grad()
def evaluate_patches(model, ds: SRDataset, device: torch.device) -> list[dict]:
    """PSNR, SSIM, MAE per patch."""
    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=model.DATA_RANGE).to(device)
    records = []

    for i in range(len(ds)):
        lr_t, hr_t, mask_t = ds[i]
        lr_t   = lr_t.unsqueeze(0).to(device)
        hr_t   = hr_t.unsqueeze(0).to(device)
        mask_t = mask_t.unsqueeze(0).to(device)

        sr_t = model(lr_t)
        sr   = model.denormalize(sr_t)
        hr   = model.denormalize(hr_t)

        sr_crop, mask_crop = model.crop_to_valid_bbox(sr, mask_t)
        hr_crop, _         = model.crop_to_valid_bbox(hr, mask_t)

        psnr = psnr_fn(sr_crop * mask_crop, hr_crop * mask_crop).item()
        ssim = ssim_fn(sr_crop, hr_crop).item()
        mae  = float((torch.abs(sr - hr) * mask_t).sum() /
                     torch.clamp(mask_t.sum(), min=1.0))

        records.append({
            "patch_id":    ds.samples.iloc[i]["patch_id"],
            "hr_image_id": ds.samples.iloc[i].get("hr_image_id", "unknown"),
            "psnr":        psnr,
            "ssim":        ssim,
            "mae_celsius": mae,
        })

    return records


# --------------- Scene-level reassembly ---------------------
@torch.no_grad()
def evaluate_scenes(
    model,
    metadata:      pd.DataFrame,
    output_dir:    str,
    hr_patch_size: int,
    device:        torch.device,
) -> list[dict]:
    """
    For each hr_image_id in metadata:
      1. Run inference on all its patches
      2. Reassemble SR canvas and HR canvas identically
         (same patch footprints, same overlap averaging)
      3. Compute global MAE / PSNR / SSIM on the reassembled pair
      4. Save SR GeoTIFF to output_dir

    No full-scene HR GeoTIFF is needed — the reference is built
    from the same patches as the prediction, making the comparison
    perfectly symmetric.
    """
    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(
        data_range=model.DATA_RANGE, kernel_size=11
    ).to(device)
    os.makedirs(output_dir, exist_ok=True)
    scene_records = []

    for scene_id, scene_patches in metadata.groupby("hr_image_id"):
        scene_patches = scene_patches.reset_index(drop=True)
        print(f"\n[INFO] Scene: {scene_id}  ({len(scene_patches)} patches)")

        # Canvas dimensions from metadata transform + patch positions
        # We don't need the full HR GeoTIFF anymore — derive canvas size
        # from the maximum patch extent recorded in metadata
        max_row = int(scene_patches["hr_row"].max()) + hr_patch_size
        max_col = int(scene_patches["hr_col"].max()) + hr_patch_size

        # Read CRS/transform from the first patch file for GeoTIFF output
        first_hr_path = scene_patches.iloc[0]["hr_path"]
        with rasterio.open(first_hr_path) as src:
            crs       = src.crs
            # Reconstruct canvas-level transform from patch metadata
            # hr_transform is stored per-patch; use the one with min row/col
            top_patch = scene_patches.loc[
                scene_patches["hr_row"].idxmin()
            ]
            t = top_patch["hr_transform"]  # [res, 0, xmin, 0, -res, ymax]
            canvas_transform = Affine(t[0], t[1], t[2], t[3], t[4], t[5])

        sr_canvas    = np.full((max_row, max_col), np.nan, dtype=np.float32)
        hr_canvas    = np.full((max_row, max_col), np.nan, dtype=np.float32)
        count_canvas = np.zeros((max_row, max_col),        dtype=np.float32)

        ds = SRDataset(scene_patches,
                       mean=model.hparams.mean,
                       std=model.hparams.std)

        for i in range(len(ds)):
            row_meta = scene_patches.iloc[i]
            hr_row   = int(row_meta["hr_row"])
            hr_col   = int(row_meta["hr_col"])

            lr_t, hr_t, _ = ds[i]
            sr_t  = model(lr_t.unsqueeze(0).to(device))
            sr_np = model.denormalize(sr_t).squeeze().cpu().numpy()
            hr_np = model.denormalize(hr_t.unsqueeze(0)).squeeze().cpu().numpy()

            r0, r1 = hr_row, min(hr_row + hr_patch_size, max_row)
            c0, c1 = hr_col, min(hr_col + hr_patch_size, max_col)
            pr, pc = r1 - r0, c1 - c0

            cnt  = count_canvas[r0:r1, c0:c1]

            # SR canvas
            prev_sr = sr_canvas[r0:r1, c0:c1]
            sr_canvas[r0:r1, c0:c1] = np.where(
                np.isnan(prev_sr),
                sr_np[:pr, :pc],
                (prev_sr * cnt + sr_np[:pr, :pc]) / (cnt + 1),
            )

            # HR canvas — identical logic, symmetric by construction
            prev_hr = hr_canvas[r0:r1, c0:c1]
            hr_canvas[r0:r1, c0:c1] = np.where(
                np.isnan(prev_hr),
                hr_np[:pr, :pc],
                (prev_hr * cnt + hr_np[:pr, :pc]) / (cnt + 1),
            )

            count_canvas[r0:r1, c0:c1] += 1

        # Valid mask: pixels covered by at least one patch in both canvases
        # By construction these are always identical, but be explicit
        valid = ~np.isnan(sr_canvas) & ~np.isnan(hr_canvas)

        if valid.sum() == 0:
            print(f"  [WARN] No valid pixels for scene {scene_id} — skipping")
            continue

        # Save SR GeoTIFF (trimmed to actual coverage)
        out_tif = os.path.join(output_dir, f"SR_{scene_id}.tif")
        sr_out  = sr_canvas.copy()
        sr_out[~valid] = -9999.0
        with rasterio.open(
            out_tif, "w",
            driver="GTiff", height=max_row, width=max_col,
            count=1, dtype=np.float32,
            crs=crs, transform=canvas_transform, nodata=-9999.0,
        ) as dst:
            dst.write(sr_out, 1)

        # Also save reassembled HR for side-by-side visual comparison
        hr_out_tif = os.path.join(output_dir, f"HR_reassembled_{scene_id}.tif")
        hr_out     = hr_canvas.copy()
        hr_out[~valid] = -9999.0
        with rasterio.open(
            hr_out_tif, "w",
            driver="GTiff", height=max_row, width=max_col,
            count=1, dtype=np.float32,
            crs=crs, transform=canvas_transform, nodata=-9999.0,
        ) as dst:
            dst.write(hr_out, 1)

        print(f"  [INFO] Saved SR  → {out_tif}")
        print(f"  [INFO] Saved HR  → {hr_out_tif}")

        # ── Global metrics on reassembled pair ──────────────────────────────
        sr_valid = sr_canvas[valid].astype(np.float32)
        hr_valid = hr_canvas[valid].astype(np.float32)

        global_mae  = float(np.abs(sr_valid - hr_valid).mean())
        global_rmse = float(np.sqrt(((sr_valid - hr_valid) ** 2).mean()))
        global_bias = float((sr_valid - hr_valid).mean())

        # PSNR on valid pixels
        sr_1d      = torch.tensor(sr_valid, device=device).unsqueeze(0).unsqueeze(0)
        hr_1d      = torch.tensor(hr_valid, device=device).unsqueeze(0).unsqueeze(0)
        scene_psnr = psnr_fn(sr_1d, hr_1d).item()

        # SSIM on tight bounding box of covered region
        rows_idx, cols_idx = np.where(valid)
        rmin, rmax = rows_idx.min(), rows_idx.max() + 1
        cmin, cmax = cols_idx.min(), cols_idx.max() + 1

        sr_crop_np = sr_canvas[rmin:rmax, cmin:cmax].copy()
        hr_crop_np = hr_canvas[rmin:rmax, cmin:cmax].copy()

        # Fill any interior NaN gaps (e.g. non-overlapping patch edges)
        # with the scene mean — prevents SSIM window from straddling nodata
        fill_val          = float(hr_valid.mean())
        interior_nan      = np.isnan(sr_crop_np) | np.isnan(hr_crop_np)
        sr_crop_np[interior_nan] = fill_val
        hr_crop_np[interior_nan] = fill_val

        sr_crop    = torch.tensor(sr_crop_np, device=device).unsqueeze(0).unsqueeze(0)
        hr_crop    = torch.tensor(hr_crop_np, device=device).unsqueeze(0).unsqueeze(0)
        scene_ssim = (
            ssim_fn(sr_crop, hr_crop).item()
            if min(rmax - rmin, cmax - cmin) >= 11
            else float("nan")
        )

        print(
            f"  MAE  : {global_mae:.4f} °C  |  RMSE : {global_rmse:.4f} °C  |"
            f"  Bias : {global_bias:+.4f} °C  |  PSNR : {scene_psnr:.2f} dB  |"
            f"  SSIM : {scene_ssim:.4f}  |  valid px : {valid.sum():,}"
        )

        scene_records.append({
            "hr_image_id":         scene_id,
            "global_mae_celsius":  global_mae,
            "global_rmse_celsius": global_rmse,
            "global_bias_celsius": global_bias,
            "scene_psnr":          scene_psnr,
            "scene_ssim":          scene_ssim,
            "n_patches":           len(scene_patches),
            "valid_pixels":        int(valid.sum()),
            "output_tif":          out_tif,
            "hr_reassembled_tif":  hr_out_tif,
        })

    return scene_records


# -------------------- Main ----------------------------

def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")
    os.makedirs("results", exist_ok=True)

    run_name = args.run_name or f"eval_{args.split}"
    wandb.init(project=args.project, name=run_name, group=args.group)

    # -Load metadata (JSON) 
    with open(args.metadata_json) as f:
        all_meta = json.load(f)
    metadata = pd.DataFrame(all_meta)
    subset   = metadata[metadata["split"] == args.split].reset_index(drop=True)
    print(f"[INFO] {len(subset)} patches in '{args.split}' split "
          f"across {subset['hr_image_id'].nunique()} scene(s)")

    # Model 
    model = load_model(args.checkpoint, device)

    # ----------------- Patch-level evaluation -------------------------------
    ds            = SRDataset(subset, mean=model.hparams.mean, std=model.hparams.std)
    patch_records = evaluate_patches(model, ds, device)

    psnr_vals = [r["psnr"]        for r in patch_records]
    ssim_vals = [r["ssim"]        for r in patch_records]
    mae_vals  = [r["mae_celsius"] for r in patch_records]

    patch_table = wandb.Table(columns=["patch_id", "hr_image_id", "psnr", "ssim", "mae_celsius"])
    for r in patch_records:
        patch_table.add_data(r["patch_id"], r["hr_image_id"],
                             r["psnr"], r["ssim"], r["mae_celsius"])

    patch_summary = {
        "patch_psnr_mean":  np.mean(psnr_vals),
        "patch_psnr_std":   np.std(psnr_vals),
        "patch_ssim_mean":  np.mean(ssim_vals),
        "patch_ssim_std":   np.std(ssim_vals),
        "patch_mae_mean":   np.mean(mae_vals),
        "patch_mae_std":    np.std(mae_vals),
    }

    print(f"\n{'─'*50}")
    print(f"  Patch PSNR : {patch_summary['patch_psnr_mean']:.2f} ± {patch_summary['patch_psnr_std']:.2f} dB")
    print(f"  Patch SSIM : {patch_summary['patch_ssim_mean']:.3f} ± {patch_summary['patch_ssim_std']:.3f}")
    print(f"  Patch MAE  : {patch_summary['patch_mae_mean']:.4f} ± {patch_summary['patch_mae_std']:.4f} °C")
    print(f"{'─'*50}\n")

    # -------------------- Scene-level evaluation ------------------------------
    scene_summary = {}

    scene_records = evaluate_scenes(
        model         = model,
        metadata      = subset,
        output_dir    = args.output_dir or "results/SR_images",
        hr_patch_size = args.hr_patch_size,
        device        = device,
    )

    if scene_records:
        scene_table = wandb.Table(
            columns=[
                "hr_image_id", "global_mae_celsius", "global_rmse_celsius",
                "global_bias_celsius", "scene_psnr", "scene_ssim",
                "n_patches", "valid_pixels"
            ]
        )
        for r in scene_records:
            scene_table.add_data(
                r["hr_image_id"],         r["global_mae_celsius"],
                r["global_rmse_celsius"], r["global_bias_celsius"],
                r["scene_psnr"],          r["scene_ssim"],
                r["n_patches"],           r["valid_pixels"],
            )

        scene_summary = {
            "scene_mae_mean":  np.mean([r["global_mae_celsius"]  for r in scene_records]),
            "scene_rmse_mean": np.mean([r["global_rmse_celsius"] for r in scene_records]),
            "scene_bias_mean": np.mean([r["global_bias_celsius"] for r in scene_records]),
            "scene_psnr_mean": np.mean([r["scene_psnr"]          for r in scene_records]),
            "scene_ssim_mean": np.mean([r["scene_ssim"]          for r in scene_records]),
        }
        wandb.log({"scene_metrics": scene_table})

        print(f"\n{'─'*50}")
        print(f"  Scene MAE  : {scene_summary['scene_mae_mean']:.4f} °C")
        print(f"  Scene RMSE : {scene_summary['scene_rmse_mean']:.4f} °C")
        print(f"  Scene Bias : {scene_summary['scene_bias_mean']:+.4f} °C")
        print(f"  Scene PSNR : {scene_summary['scene_psnr_mean']:.2f} dB")
        print(f"  Scene SSIM : {scene_summary['scene_ssim_mean']:.4f}")
        print(f"{'─'*50}\n")

    wandb.log({"patch_metrics": patch_table, **patch_summary, **scene_summary})
    wandb.finish()
    print(f"[INFO] Completed. Results logged to W&B run '{run_name}'")


# -------------------- CLI ----------------------------
def parse_args():
    p = argparse.ArgumentParser(
        description="Evaluate TIR SR model — patch metrics + full scene reassembly"
    )
    p.add_argument("--checkpoint",    required=True,
                   help="Local .ckpt or 'wandb:entity/project/model-ID:best'")
    p.add_argument("--metadata_json", default="data/full_metadata.json")
    p.add_argument("--split",         default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--output_dir",    default=None,
                   help="Where to save SR + HR reassembled GeoTIFFs")
    p.add_argument("--hr_patch_size", type=int, default=512)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--project",       default="TIR_sisr")
    p.add_argument("--group",         default=None)
    p.add_argument("--run_name",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()