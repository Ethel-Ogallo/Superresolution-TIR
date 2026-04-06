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
    metadata:     pd.DataFrame,
    hr_full_dir:  str,
    output_dir:   str,
    hr_patch_size: int,
    device:       torch.device,
) -> list[dict]:
    """
    For each hr_image_id in metadata:
      1. Run inference on all its patches
      2. Place SR outputs at (hr_row, hr_col) in a canvas matching the full HR image
      3. Average overlapping regions
      4. Compute global MAE vs full HR GeoTIFF
      5. Save SR GeoTIFF to output_dir

    Returns list of per-scene metric dicts.
    """
    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    os.makedirs(output_dir, exist_ok=True)
    scene_records = []

    for scene_id, scene_patches in metadata.groupby("hr_image_id"):
        scene_patches = scene_patches.reset_index(drop=True)
        print(f"\n[INFO] Scene: {scene_id}  ({len(scene_patches)} patches)")

        # Find full HR image 
        hr_full_path = os.path.join(hr_full_dir, f"{scene_id}.tif")
        if not os.path.exists(hr_full_path):
            print(f"  [WARN] Full HR not found at {hr_full_path} — skipping scene")
            continue

        with rasterio.open(hr_full_path) as src:
            full_h, full_w = src.height, src.width
            transform      = src.transform
            crs            = src.crs
            hr_nodata      = src.nodata if src.nodata is not None else -9999.0
            hr_full_arr    = src.read(1).astype(np.float32)

        sr_canvas    = np.full((full_h, full_w), np.nan, dtype=np.float32)
        count_canvas = np.zeros((full_h, full_w), dtype=np.float32)

        # Dataset for this scene's patches only
        ds = SRDataset(scene_patches,
                       mean=model.hparams.mean,
                       std=model.hparams.std)

        for i in range(len(ds)):
            row_meta = scene_patches.iloc[i]
            hr_row   = int(row_meta["hr_row"])
            hr_col   = int(row_meta["hr_col"])

            lr_t, _, _ = ds[i]
            sr_t  = model(lr_t.unsqueeze(0).to(device))
            sr_np = model.denormalize(sr_t).squeeze().cpu().numpy()

            r0 = hr_row
            r1 = min(hr_row + hr_patch_size, full_h)
            c0 = hr_col
            c1 = min(hr_col + hr_patch_size, full_w)
            pr, pc = r1 - r0, c1 - c0

            # Running average (handles 50% overlap stride)
            prev = sr_canvas[r0:r1, c0:c1]
            cnt  = count_canvas[r0:r1, c0:c1]
            sr_canvas[r0:r1, c0:c1] = np.where(
                np.isnan(prev),
                sr_np[:pr, :pc],
                (prev * cnt + sr_np[:pr, :pc]) / (cnt + 1),
            )
            count_canvas[r0:r1, c0:c1] += 1

        # Fill unvisited pixels with nodata
        sr_canvas[np.isnan(sr_canvas)] = hr_nodata

        # Save SR GeoTIFF 
        out_tif = os.path.join(output_dir, f"SR_{scene_id}.tif")
        with rasterio.open(
            out_tif, "w",
            driver="GTiff", height=full_h, width=full_w,
            count=1, dtype=np.float32,
            crs=crs, transform=transform, nodata=hr_nodata,
        ) as dst:
            dst.write(sr_canvas, 1)
        print(f"  [INFO] Saved → {out_tif}")

        # Global MAE 
        gt_nodata = hr_nodata
        valid = (sr_canvas != hr_nodata)
        if gt_nodata is not None:
            valid &= (hr_full_arr != gt_nodata)
        else:
            valid &= ~np.isnan(hr_full_arr)

        if valid.sum() == 0:
            print(f"  [WARN] No valid pixels for scene {scene_id}")
            continue

        global_mae = float(np.abs(sr_canvas[valid] - hr_full_arr[valid]).mean())

        #  Scene PSNR / SSIM on valid region 
        # Extract only valid pixels as 1D then reshape to minimal 2D patch
        # to avoid SSIM NaN from nodata-filled windows in the full scene canvas
        sr_valid_px = sr_canvas[valid].astype(np.float32)
        hr_valid_px = hr_full_arr[valid].astype(np.float32)

        # PSNR on valid pixels directly
        sr_1d = torch.tensor(sr_valid_px, device=device).unsqueeze(0).unsqueeze(0)
        hr_1d = torch.tensor(hr_valid_px, device=device).unsqueeze(0).unsqueeze(0)
        scene_psnr = psnr_fn(sr_1d, hr_1d).item()

        # SSIM needs 2D spatial context — reshape valid pixels into a 2D strip
        # Use patch-level SSIM averaged across patches as scene-level SSIM
        # (full-scene SSIM on sparse river strip is not meaningful)
        patch_ssims = []
        ds_scene = SRDataset(scene_patches, mean=model.hparams.mean, std=model.hparams.std)
        ssim_patch_fn = StructuralSimilarityIndexMeasure(data_range=model.DATA_RANGE).to(device)

        for i in range(len(ds_scene)):
            lrt, hrt, mkt = ds_scene[i]
            with torch.no_grad():
                srt = model(lrt.unsqueeze(0).to(device))
            sr_p = model.denormalize(srt)
            hr_p = model.denormalize(hrt.unsqueeze(0).to(device))
            mk_p = mkt.unsqueeze(0).to(device)
            sr_c, mk_c = model.crop_to_valid_bbox(sr_p, mk_p)
            hr_c, _    = model.crop_to_valid_bbox(hr_p, mk_p)
            if sr_c.shape[-1] >= 11 and sr_c.shape[-2] >= 11:
                patch_ssims.append(ssim_patch_fn(sr_c, hr_c).item())

        scene_ssim = float(np.mean(patch_ssims)) if patch_ssims else float("nan")

        print(f"  MAE : {global_mae:.4f} °C  |  PSNR : {scene_psnr:.2f} dB  |  SSIM : {scene_ssim:.3f}")

        scene_records.append({
            "hr_image_id":      scene_id,
            "global_mae_celsius": global_mae,
            "scene_psnr":       scene_psnr,
            "scene_ssim":       scene_ssim,
            "n_patches":        len(scene_patches),
            "valid_pixels":     int(valid.sum()),
            "output_tif":       out_tif,
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
    if args.hr_full_dir:
        scene_records = evaluate_scenes(
            model         = model,
            metadata      = subset,
            hr_full_dir   = args.hr_full_dir,
            output_dir    = args.output_dir or "results/SR_images",
            hr_patch_size = args.hr_patch_size,
            device        = device,
        )

        if scene_records:
            scene_table = wandb.Table(
                columns=["hr_image_id", "global_mae_celsius",
                         "scene_psnr", "scene_ssim", "n_patches"]
            )
            for r in scene_records:
                scene_table.add_data(
                    r["hr_image_id"], r["global_mae_celsius"],
                    r["scene_psnr"],  r["scene_ssim"], r["n_patches"]
                )

            scene_summary = {
                "global_mae_mean":  np.mean([r["global_mae_celsius"] for r in scene_records]),
                "scene_psnr_mean":  np.mean([r["scene_psnr"]         for r in scene_records]),
                "scene_ssim_mean":  np.mean([r["scene_ssim"]         for r in scene_records]),
            }
            wandb.log({"scene_metrics": scene_table})

            print(f"\n{'─'*50}")
            print(f"  Global MAE  : {scene_summary['global_mae_mean']:.4f} °C")
            print(f"  Scene PSNR  : {scene_summary['scene_psnr_mean']:.2f} dB")
            print(f"  Scene SSIM  : {scene_summary['scene_ssim_mean']:.3f}")
            print(f"{'─'*50}\n")
    else:
        print("[WARN] --hr_full_dir not provided — skipping scene-level evaluation")

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
    p.add_argument("--metadata_json", default="data/full_metadata.json",
                   help="Global metadata JSON (from create_metadata.py)")
    p.add_argument("--split",         default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--hr_full_dir",   default=None,
                   help="Folder with full HR GeoTIFFs named {hr_image_id}.tif")
    p.add_argument("--output_dir",    default=None,
                   help="Where to save SR full images (default: results/SR_images/)")
    p.add_argument("--hr_patch_size", type=int, default=512)
    p.add_argument("--num_workers",   type=int, default=2)
    p.add_argument("--project",       default="TIR_sisr")
    p.add_argument("--group",         default=None)
    p.add_argument("--run_name",      default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()