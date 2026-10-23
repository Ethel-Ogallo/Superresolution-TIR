"""Single-batch FastDiffSR inference smoke test. Does not train or update weights."""

import argparse
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader

# Project directories
PROJECT_ROOT = Path(__file__).resolve().parents[3]  # Superresolution-TIR
SISR_ROOT = PROJECT_ROOT / "SISR"
SCRIPTS_ROOT = SISR_ROOT / "scripts"

# Allow imports from scripts/
sys.path.insert(0, str(SCRIPTS_ROOT))

from utils.dataset import SRDataset
from models.fastdiffsr import FastDiffSRModule

# Data and pretrained checkpoint (outside SISR/)
PATCHES_DIR = PROJECT_ROOT / "data" / "processed" / "patches"
STATS_PATH = PATCHES_DIR / "stats.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"


def describe(name, x, mask=None):
    x = x.detach()
    valid = torch.isfinite(x)

    if mask is not None:
        valid = valid & mask.bool().expand_as(x)

    values = x[valid]
    if values.numel() == 0:
        print(f"{name}: no valid pixels")
        return

    print(
        f"{name}: shape={tuple(x.shape)}, "
        f"min={values.min().item():.3f} °C, "
        f"median={values.median().item():.3f} °C, "
        f"max={values.max().item():.3f} °C, "
        f"finite={torch.isfinite(x).all().item()}"
    )


def masked_mae(pred, target, mask):
    valid = mask.bool() & torch.isfinite(pred) & torch.isfinite(target)

    if not valid.any():
        return float("nan"), 0

    return (pred[valid] - target[valid]).abs().mean().item(), int(valid.sum())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", type=int, default=1)
    args = parser.parse_args()

    print(f"Repository: {PROJECT_ROOT}")
    print(f"Checkpoint: {CHECKPOINT_PATH}")

    for path in (STATS_PATH, CHECKPOINT_PATH):
        if not path.exists():
            raise FileNotFoundError(f"Required file not found: {path}")

    with open(STATS_PATH) as f:
        stats = json.load(f)

    # One deterministic validation batch; no training augmentation.
    dataset = SRDataset(
        split="val",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        repeat_channels=True,
        transform=None,
        use_water_mask=True,
    )

    if len(dataset) == 0:
        raise RuntimeError("Validation dataset is empty.")

    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=0,
    )
    batch = next(iter(loader))

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")
    print(f"Validation samples: {len(dataset)}")
    print(f"Sample: {batch['fname'][0] if isinstance(batch['fname'], (list, tuple)) else batch['fname']}")

    # Instantiate the pretrained model; no optimizer or training is used.
    model = FastDiffSRModule(
        stats_path=str(STATS_PATH),
        checkpoint_path=str(CHECKPOINT_PATH),
    ).to(device)
    model.eval()

    batch = {
        key: value.to(device) if torch.is_tensor(value) else value
        for key, value in batch.items()
    }

    with torch.no_grad():
        # Adapter prepares the shared-scale HR target and upsampled LR condition.
        lr_up, hr_scaled, hr_mask = model._prepare_batch(batch)

        print(f"Condition shape: {tuple(lr_up.shape)}")
        print(f"HR target shape: {tuple(hr_scaled.shape)}")

        if lr_up.shape[-2:] != (256, 256):
            raise RuntimeError(f"Unexpected condition size: {lr_up.shape[-2:]}")

        # Bicubic baseline in physical Celsius.
        lr_c = (
            batch["lr"][:, 0:1] * model.lr_std + model.lr_mean
        )
        hr_c = (
            batch["hr"][:, 0:1] * model.hr_std + model.hr_mean
        )
        bicubic_c = F.interpolate(
            lr_c,
            size=hr_c.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        # Pretrained diffusion inference.
        # This can take longer than a feed-forward SR model.
        sr_scaled = model(lr_up)

        if sr_scaled.ndim != 4:
            raise RuntimeError(
                f"Expected 4D output [B,C,H,W], got {tuple(sr_scaled.shape)}"
            )

        if sr_scaled.shape[-2:] != hr_c.shape[-2:]:
            raise RuntimeError(
                f"Output size {sr_scaled.shape[-2:]} does not match "
                f"HR size {hr_c.shape[-2:]}"
            )

        if not torch.isfinite(sr_scaled).all():
            raise RuntimeError("FastDiffSR output contains NaN or infinity.")

        # Convert the model output back to Celsius.
        sr_c = model._to_celsius(sr_scaled[:, 0:1])
        sr_c = sr_c.float()
        hr_c = hr_c.float()
        bicubic_c = bicubic_c.float()

        valid_mask = hr_mask.bool()
        water_mask = batch.get("water_mask")

        if water_mask is not None:
            water_mask = water_mask.bool() & valid_mask
        else:
            print("WARNING: no water mask found; reporting valid-pixel MAE only.")

        print("\n--- Output diagnostics ---")
        describe("HR reference", hr_c, valid_mask)
        describe("Bicubic baseline", bicubic_c, valid_mask)
        describe("FastDiffSR prediction", sr_c, valid_mask)

        print("\n--- MAE (°C) ---")
        mae, n = masked_mae(bicubic_c, hr_c, valid_mask)
        print(f"Bicubic valid-pixel MAE: {mae:.4f} (n={n})")

        mae, n = masked_mae(sr_c, hr_c, valid_mask)
        print(f"FastDiffSR valid-pixel MAE: {mae:.4f} (n={n})")

        if water_mask is not None:
            mae, n = masked_mae(bicubic_c, hr_c, water_mask)
            print(f"Bicubic water MAE: {mae:.4f} (n={n})")

            mae, n = masked_mae(sr_c, hr_c, water_mask)
            print(f"FastDiffSR water MAE: {mae:.4f} (n={n})")

        print("\nSmoke test completed.")
        print(
            "Reminder: successful inference does not prove that the "
            "normalization matches the checkpoint's original training setup."
        )


if __name__ == "__main__":
    main()
