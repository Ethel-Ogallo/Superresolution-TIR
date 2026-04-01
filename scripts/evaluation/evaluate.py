# ----------------- Imports -----------------
import argparse
import os
import torch
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure
from scripts.utils.dataset import SRDataset
import rasterio

# ----------------- Model loader -----------------
def load_model(ckpt_path: str, device: torch.device):
    from scripts.models.edsr import EDSRModule
    REGISTRY = {"EDSRModule": EDSRModule}

    if ckpt_path.startswith("wandb:"):
        api = wandb.Api()
        art = api.artifact(ckpt_path.removeprefix("wandb:"))
        local = art.download(root="/tmp/ckpt")
        ckpt_path = next(os.path.join(local, f) for f in os.listdir(local) if f.endswith(".ckpt"))

    raw = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls_name = raw.get("hyper_parameters", {}).get("model_class", None)

    if cls_name and cls_name in REGISTRY:
        model = REGISTRY[cls_name].load_from_checkpoint(ckpt_path, map_location=device)
        return model.eval().to(device), os.path.basename(ckpt_path)

    for name, cls in REGISTRY.items():
        try:
            model = cls.load_from_checkpoint(ckpt_path, map_location=device)
            return model.eval().to(device), os.path.basename(ckpt_path)
        except Exception:
            continue

    raise ValueError("Could not infer model class. Register it in evaluate.py::REGISTRY.")

# ----------------- Per-patch evaluation -----------------
@torch.no_grad()
def run_patch_metrics(model, loader, device):
    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=model.DATA_RANGE).to(device)
    records = []

    for i, (lr_t, hr_t, mask_t) in enumerate(loader):
        lr_t, hr_t, mask_t = lr_t.to(device), hr_t.to(device), mask_t.to(device)
        sr_t = model(lr_t)

        sr = model.denormalize(sr_t)
        hr = model.denormalize(hr_t)

        sr_crop, mask_crop = model.crop_to_valid_bbox(sr, mask_t)
        hr_crop, _ = model.crop_to_valid_bbox(hr, mask_t)

        psnr = psnr_fn(sr_crop * mask_crop, hr_crop * mask_crop).item()
        ssim = ssim_fn(sr_crop, hr_crop).item()

        records.append({"patch": i, "psnr": psnr, "ssim": ssim})

    return records

# ----------------- Full-scene evaluation (fixed) -----------------
@torch.no_grad()
def run_full_scene_metrics(model, metadata_df, device):
    """
    Reconstruct full HR scenes from patches, run model, compute metrics safely.
    Handles missing patches (e.g., cloudy patches) by masking only valid pixels.

    Args:
        model       : SR model
        metadata_df : DataFrame with columns 
                      ['hr_image_id','hr_path','lr_path','hr_row','hr_col','lr_row','lr_col']
        device      : torch.device

    Returns:
        dict of scene_id -> metrics (mae, psnr, ssim, num_patches)
    """
    from torchmetrics import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=model.DATA_RANGE).to(device)

    scene_metrics = {}

    for scene_id, patches in metadata_df.groupby("hr_image_id"):

        # Open first HR patch to get full image shape
        hr_first_path = patches.iloc[0]["hr_path"]
        with rasterio.open(hr_first_path) as src:
            full_hr_shape = (src.height, src.width)
            full_hr = np.zeros(full_hr_shape, dtype=np.float32)
            patch_counter = np.zeros(full_hr_shape, dtype=np.uint8)

        # Assemble HR ground truth
        for _, patch_info in patches.iterrows():
            hr_patch_path = patch_info["hr_path"]
            hr_row, hr_col = patch_info["hr_row"], patch_info["hr_col"]

            with rasterio.open(hr_patch_path) as src:
                hr_patch = src.read(1).astype(np.float32)

            h, w = hr_patch.shape
            max_h = min(h, full_hr.shape[0] - hr_row)
            max_w = min(w, full_hr.shape[1] - hr_col)
            if max_h <= 0 or max_w <= 0:
                continue

            full_hr[hr_row:hr_row+max_h, hr_col:hr_col+max_w] += hr_patch[:max_h, :max_w]
            patch_counter[hr_row:hr_row+max_h, hr_col:hr_col+max_w] += 1

        # Only keep valid pixels
        valid_mask = patch_counter > 0
        full_hr = np.divide(full_hr, patch_counter, out=np.zeros_like(full_hr), where=valid_mask)

        # Reconstruct full SR scene
        full_sr = np.zeros_like(full_hr)
        sr_counter = np.zeros_like(full_hr, dtype=np.uint8)

        for _, patch_info in patches.iterrows():
            lr_patch_path = patch_info["lr_path"]
            hr_row, hr_col = patch_info["hr_row"], patch_info["hr_col"]

            with rasterio.open(lr_patch_path) as src:
                lr_patch = src.read(1).astype(np.float32)

            lr_t = torch.tensor(lr_patch[None, None, :, :], device=device)
            sr_patch = model.denormalize(model(lr_t)).cpu().numpy()[0,0]

            h, w = sr_patch.shape
            max_h = min(h, full_sr.shape[0] - hr_row)
            max_w = min(w, full_sr.shape[1] - hr_col)
            if max_h <= 0 or max_w <= 0:
                continue

            full_sr[hr_row:hr_row+max_h, hr_col:hr_col+max_w] += sr_patch[:max_h, :max_w]
            sr_counter[hr_row:hr_row+max_h, hr_col:hr_col+max_w] += 1

        full_sr = np.divide(full_sr, sr_counter, out=np.zeros_like(full_sr), where=sr_counter>0)

        # Convert to tensors and mask only valid pixels
        valid_mask_tensor = torch.tensor(valid_mask[None, None], device=device, dtype=torch.bool)
        full_hr_tensor = torch.tensor(full_hr[None, None], device=device, dtype=torch.float32)
        full_sr_tensor = torch.tensor(full_sr[None, None], device=device, dtype=torch.float32)

        full_hr_valid = torch.masked_select(full_hr_tensor, valid_mask_tensor)
        full_sr_valid = torch.masked_select(full_sr_tensor, valid_mask_tensor)

        # Compute metrics safely
        if full_hr_valid.numel() > 0:
            mae  = torch.mean(torch.abs(full_sr_valid - full_hr_valid)).item()
            psnr = psnr_fn(full_sr_valid.unsqueeze(0).unsqueeze(0),
                           full_hr_valid.unsqueeze(0).unsqueeze(0)).item()
            ssim = ssim_fn(full_sr_valid.unsqueeze(0).unsqueeze(0),
                           full_hr_valid.unsqueeze(0).unsqueeze(0)).item()
        else:
            mae = psnr = ssim = float('nan')

        scene_metrics[scene_id] = {
            "mae": mae,
            "psnr": psnr,
            "ssim": ssim,
            "num_patches": len(patches),
            "valid_pixel_percent": 100 * valid_mask.sum() / valid_mask.size
        }

    return scene_metrics

# ----------------- Main -----------------
def main():
    args = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

    run_name = args.run_name or f"eval_{args.split}"
    run = wandb.init(project=args.project, name=run_name, group=args.group)

    # Load metadata
    metadata = pd.read_json(args.metadata_json, orient="records")

    # ---------------- Patch-level ----------------
    subset = metadata[metadata["split"] == args.split].reset_index(drop=True)
    print(f"[INFO] Evaluating {len(subset)} patches from split '{args.split}'")
    model, ckpt_name = load_model(args.checkpoint, device)
    ds = SRDataset(subset, mean=model.hparams.mean, std=model.hparams.std)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    patch_records = run_patch_metrics(model, loader, device)

    # Patch-level table
    patch_table = wandb.Table(columns=["patch", "psnr", "ssim"])
    for r in patch_records:
        patch_table.add_data(r["patch"], r["psnr"], r["ssim"])
    wandb.log({"patch_metrics": patch_table})

    patch_psnr = [r["psnr"] for r in patch_records]
    patch_ssim = [r["ssim"] for r in patch_records]
    wandb.log({"patch_psnr_mean": np.mean(patch_psnr),
               "patch_psnr_std": np.std(patch_psnr),
               "patch_ssim_mean": np.mean(patch_ssim),
               "patch_ssim_std": np.std(patch_ssim)})

    # ---------------- Full-scene ----------------
    print("[INFO] Computing full-scene metrics on all patches...")
    scene_metrics = run_full_scene_metrics(model, metadata, device)

    # Full-scene table
    scene_table = wandb.Table(columns=["hr_image_id", "mae", "psnr", "ssim", "num_patches"])
    for hr_id, metrics in scene_metrics.items():
        scene_table.add_data(hr_id, metrics["mae"], metrics["psnr"], metrics["ssim"], metrics["num_patches"])
        wandb.log({f"{hr_id}_mae": metrics["mae"],
                   f"{hr_id}_psnr": metrics["psnr"],
                   f"{hr_id}_ssim": metrics["ssim"]})

    wandb.log({"full_scene_metrics": scene_table})

    wandb.finish()
    print("[INFO] Completed evaluation.")

# ----------------- CLI -----------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate any TIR SR checkpoint")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--metadata_json", default="full_metadata.json")
    p.add_argument("--split", default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--project", default="TIR_sisr")
    p.add_argument("--run_name", default=None)
    p.add_argument("--group", default="Evaluation")
    return p.parse_args()

if __name__ == "__main__":
    main()