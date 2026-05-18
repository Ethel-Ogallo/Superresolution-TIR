# scripts/evaluation/eval.py
"""
eval.py — Phase 1 and Phase 2 evaluation for all SR models.

Phase 1 (inference): python -m scripts.evaluation.eval --model edsr --phase 1
Phase 2 (inference after fine-tuning): python -m scripts.evaluation.eval --model edsr --phase 2 --checkpoint path/to/ckpt.ckpt
Run all: python -m scripts.evaluation.eval --all --phase 1
"""

import argparse
import json
import torch
from pathlib import Path
from torch.utils.data import DataLoader
from lightning.pytorch import Trainer

BASE        = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH  = PATCHES_DIR / "stats.json"
PRETRAINED  = BASE / "data/pretrained"
RESULTS_DIR = BASE / "results"

with open(STATS_PATH) as f:
    stats = json.load(f)

HR_MEAN    = stats["hr"]["mean"]
HR_STD     = stats["hr"]["std"]
DATA_RANGE = stats["hr_data_range"]
DATA_MIN   = stats["hr_percentiles"]["p1"]


def build_model(model_name, phase, checkpoint=None):
    # Shared normalization stats 
    common_norm = dict(
        hr_mean=HR_MEAN,
        hr_std=HR_STD,
    )

    # Classical SR models need data_range
    sr_common = dict(
        **common_norm,
        data_range=DATA_RANGE,
        data_min=DATA_MIN,
        phase = phase,
    )

    if model_name == "edsr":
        from scripts.models.edsr import EDSRModule

        if checkpoint:
            return EDSRModule.load_from_checkpoint(checkpoint, **sr_common)

        return EDSRModule(
            pretrained_path=str(PRETRAINED / "EDSR_baseline_x4.pth"),
            **sr_common
        )

    elif model_name == "swinir":
        from scripts.models.swinir import SwinIRModule

        if checkpoint:
            return SwinIRModule.load_from_checkpoint(checkpoint, **sr_common)

        return SwinIRModule(
            pretrained_path=str(PRETRAINED / "SwinIR_classical_x4.pth"),
            **sr_common
        )

    elif model_name == "hat":
        from scripts.models.hat import HATModule

        if checkpoint:
            return HATModule.load_from_checkpoint(checkpoint, **sr_common)

        return HATModule(
            pretrained_path=str(PRETRAINED / "HAT_imagenet_x4.pth"),
            **sr_common
        )

    elif model_name == "realesrgan":
        from scripts.models.realesrgan import RealESRGANModule

        if checkpoint:
            return RealESRGANModule.load_from_checkpoint(checkpoint, **sr_common)

        return RealESRGANModule(
            pretrained_path=str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
            **sr_common
        )

    elif model_name == "resshift":
        from scripts.models.resshift import ResShiftModule

        if checkpoint:
            return ResShiftModule.load_from_checkpoint(checkpoint)

        return ResShiftModule(
            pretrained_path=str(PRETRAINED / "ResShift_x4.pth"),
            ae_path=str(PRETRAINED / "autoencoder_vq_f4.pth"),
            hr_mean=HR_MEAN,
            hr_std=HR_STD,
        )

    raise ValueError(f"Unknown model: {model_name}")


def run_inference(model_name, phase, checkpoint=None, use_water_metrics=False, group=None):

    print(f"\n{'═'*60}")
    print(f"Model : {model_name.upper()}")
    print(f"Phase : {phase}")
    print(f"Water metrics : {use_water_metrics}")
    print(f"{'═'*60}")

    import wandb
    from lightning.pytorch.loggers import WandbLogger
    from scripts.utils.dataset import SRDataset

    test_ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_water_mask=use_water_metrics,
        repeat_channels=True,
        transform=None,
    )

    test_loader = DataLoader(test_ds, batch_size=4, shuffle=False)

    model = build_model(model_name, phase, checkpoint)

    wandb_logger = WandbLogger(
        project="TIR_sisr",
        name=args.run_name or f"{model_name}_phase{phase}_eval",
        group=group or model_name,
        log_model=False
    )

    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        precision="32",
        logger=wandb_logger,
        enable_checkpointing=False
    )

    results = trainer.test(model, dataloaders=test_loader)

    if results:
        metrics = results[0]

        wandb.log({
            "test_psnr": metrics.get("test_psnr", 0),
            "test_ssim": metrics.get("test_ssim", 0),
            "test_mae": metrics.get("test_mae", 0),
            "test_rmse": metrics.get("test_rmse", 0),
            "test_water_mae": metrics.get("test_water_mae", None),
            "test_water_rmse": metrics.get("test_water_rmse", None),
        })

    wandb.finish()
    return results


def run_all(phase, checkpoint_dir=None):
    models      = ["edsr", "swinir", "hat", "realesrgan", "resshift"]
    all_results = {}

    for model_name in models:
        ckpt = None
        if checkpoint_dir:
            ckpt_path = Path(checkpoint_dir) / f"{model_name}_best.ckpt"
            ckpt = str(ckpt_path) if ckpt_path.exists() else None
        try:
            results = run_inference(model_name, phase, ckpt)
            if results:
                all_results[model_name] = results[0]
        except Exception as e:
            print(f"[ERROR] {model_name} failed: {e}")
            all_results[model_name] = {"error": str(e)}

    # Comparison table 
    print(f"\n{'═'*60}")
    print(f"PHASE {phase} BENCHMARK SUMMARY")
    print(f"{'═'*60}")
    print(f"{'Model':<15} {'PSNR':>8} {'SSIM':>8} {'MAE':>8} {'RMSE':>8}")
    print(f"{'─'*15} {'─'*8} {'─'*8} {'─'*8} {'─'*8}")

    for name, metrics in all_results.items():
        if "error" in metrics:
            print(f"{name:<15} {'ERROR':>8}")
            continue
        print(
            f"{name:<15} "
            f"{metrics.get('test_psnr', 0):>8.4f} "
            f"{metrics.get('test_ssim', 0):>8.4f} "
            f"{metrics.get('test_mae',  0):>8.4f} "
            f"{metrics.get('test_rmse', 0):>8.4f}"
        )

    out = RESULTS_DIR / f"phase{phase}" / f"phase{phase}_comparison.json"
    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"\nComparison saved to {out}")


if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        choices=["edsr","swinir","hat","realesrgan","resshift"])
    parser.add_argument("--phase", type=int, default=1, choices=[1,2])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--use_water_metrics", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--project", default="TIR_sisr")

    args = parser.parse_args()

    if args.all:
        run_all(args.phase, args.checkpoint_dir)

    elif args.model:
        run_inference(
            args.model,
            args.phase,
            args.checkpoint,
            use_water_metrics=args.use_water_metrics,
            group=args.group
        )

    else:
        parser.print_help()