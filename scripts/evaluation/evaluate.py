"""
evaluate.py — Model-agnostic evaluation for TIR SR models.

Logs per-patch and aggregate metrics to W&B.
No local disk usage beyond /tmp (cluster-friendly).
"""

import argparse
import os

import torch
import numpy as np
import pandas as pd
import wandb
from torch.utils.data import DataLoader
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.dataset import SRDataset


# ------------ Model registry -----------------------------------------------
def load_model(ckpt_path: str, device: torch.device):
    from scripts.models.edsr import EDSRModule
    # from scripts.models.swinir import SwinIRModule
    # from scripts.models.hat    import HATModule
    REGISTRY = {
        "EDSRModule": EDSRModule,
    }

    # Download from W&B artifact if needed
    if ckpt_path.startswith("wandb:"):
        api   = wandb.Api()
        art   = api.artifact(ckpt_path.removeprefix("wandb:"))
        local = art.download(root="/tmp/ckpt")
        ckpt_path = next(
            os.path.join(local, f) for f in os.listdir(local) if f.endswith(".ckpt")
        )
        print(f"[INFO] Downloaded checkpoint → {ckpt_path}")

    # Try to infer class from hparams
    raw      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cls_name = raw.get("hyper_parameters", {}).get("model_class", None)

    if cls_name and cls_name in REGISTRY:
        model = REGISTRY[cls_name].load_from_checkpoint(ckpt_path, map_location=device)
        print(f"[INFO] Loaded {cls_name}")
        return model.eval().to(device), os.path.basename(ckpt_path)

    # Fallback — try each registered class
    for name, cls in REGISTRY.items():
        try:
            model = cls.load_from_checkpoint(ckpt_path, map_location=device)
            print(f"[INFO] Loaded {name} (fallback)")
            return model.eval().to(device), os.path.basename(ckpt_path)
        except Exception:
            continue

    raise ValueError("Could not infer model class. Register it in evaluate.py::REGISTRY.")


# --------------- Per-patch metrics -------------------------

@torch.no_grad()
def run_evaluation(model, loader, device) -> list[dict]:
    """Compute per-patch PSNR and SSIM. Returns list of dicts."""
    psnr_fn = PeakSignalNoiseRatio(data_range=model.DATA_RANGE).to(device)
    ssim_fn = StructuralSimilarityIndexMeasure(data_range=model.DATA_RANGE).to(device)

    records = []
    for i, (lr_t, hr_t, mask_t) in enumerate(loader):
        lr_t, hr_t, mask_t = lr_t.to(device), hr_t.to(device), mask_t.to(device)
        sr_t = model(lr_t)

        sr = model.denormalize(sr_t)
        hr = model.denormalize(hr_t)

        sr_crop, mask_crop = model.crop_to_valid_bbox(sr, mask_t)
        hr_crop, _         = model.crop_to_valid_bbox(hr, mask_t)

        psnr = psnr_fn(sr_crop * mask_crop, hr_crop * mask_crop).item()
        ssim = ssim_fn(sr_crop, hr_crop).item()

        records.append({"patch": i, "psnr": psnr, "ssim": ssim})

    return records


# ------------------ Main --------------------------
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

    run_name = args.run_name or f"eval_{args.split}"
    run = wandb.init(project=args.project, 
                     name=args.run_name, 
                     group=args.group
                     )

    # Data
    metadata = pd.read_csv(args.metadata_csv)
    subset   = metadata[metadata["split"] == args.split].reset_index(drop=True)
    print(f"[INFO] Evaluating {len(subset)} patches from '{args.split}' split")

    # Model 
    model, ckpt_name = load_model(args.checkpoint, device)
    mean, std        = model.hparams.mean, model.hparams.std

    ds     = SRDataset(subset, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Evaluate 
    records = run_evaluation(model, loader, device)

    psnr_vals = [r["psnr"] for r in records]
    ssim_vals = [r["ssim"] for r in records]

    # Per-patch table
    table = wandb.Table(columns=["patch", "psnr", "ssim"])
    for r in records:
        table.add_data(r["patch"], r["psnr"], r["ssim"])

    # Aggregate summary
    summary = {
        "test_psnr_mean": np.mean(psnr_vals),
        "test_psnr_std":  np.std(psnr_vals),
        "test_ssim_mean": np.mean(ssim_vals),
        "test_ssim_std":  np.std(ssim_vals),
    }

    wandb.log({"per_patch_metrics": table, **summary})

    print(f"\n{'─'*40}")
    print(f"  PSNR : {summary['test_psnr_mean']:.2f} ± {summary['test_psnr_std']:.2f} dB")
    print(f"  SSIM : {summary['test_ssim_mean']:.3f} ± {summary['test_ssim_std']:.3f}")
    print(f"{'─'*40}\n")

    wandb.finish()
    print(f"[INFO] Done. Metrics logged to W&B run '{run_name}'")


# ----------------- CLI ----------------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate any TIR SR checkpoint")
    p.add_argument("--checkpoint",   required=True,
                   help="Local .ckpt path or 'wandb:entity/project/model-ID:best'")
    p.add_argument("--metadata_csv", default="metadata.csv")
    p.add_argument("--split",        default="test",
                   choices=["train", "val", "test"])
    p.add_argument("--num_workers",  type=int, default=2)
    p.add_argument("--project",      default="TIR_sisr")
    p.add_argument("--run_name",     default=None)
    p.add_argument("--group",        default="Evaluation")
    return p.parse_args()


if __name__ == "__main__":
    main()