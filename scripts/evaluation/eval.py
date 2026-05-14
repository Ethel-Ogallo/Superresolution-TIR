# scripts/evaluation/eval.py

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


# -----------------------------
# MODEL BUILDER
# -----------------------------
def build_model(model_name, phase, checkpoint=None):

    common_norm = dict(
        hr_mean=HR_MEAN,
        hr_std=HR_STD,
        data_range=DATA_RANGE,
        phase=phase,
    )

    # ---------------- SWINIR ----------------
    if model_name == "swinir":
        from scripts.models.swinir import SwinIRModule

        model = SwinIRModule(
            pretrained_path=str(PRETRAINED / "SwinIR_classical_x4.pth"),
            **common_norm
        )

        if checkpoint:
            print(f"[INFO] Loading checkpoint: {checkpoint}")
            ckpt = torch.load(checkpoint, map_location="cpu")

            # SAFE LOAD: avoid proj mismatch crashes
            state = ckpt.get("state_dict", ckpt.get("params", ckpt))

            missing, unexpected = model.load_state_dict(state, strict=False)

            print(f"[INFO] Missing keys: {len(missing)}")
            print(f"[INFO] Unexpected keys: {len(unexpected)}")

        return model


    # ---------------- EDSR ----------------
    elif model_name == "edsr":
        from scripts.models.edsr import EDSRModule

        return EDSRModule.load_from_checkpoint(
            checkpoint,
            **common_norm
        ) if checkpoint else EDSRModule(
            pretrained_path=str(PRETRAINED / "EDSR_baseline_x4.pth"),
            **common_norm
        )


    # ---------------- HAT ----------------
    elif model_name == "hat":
        from scripts.models.hat import HATModule

        return HATModule.load_from_checkpoint(
            checkpoint,
            **common_norm
        ) if checkpoint else HATModule(
            pretrained_path=str(PRETRAINED / "HAT_imagenet_x4.pth"),
            **common_norm
        )


    # ---------------- RealESRGAN ----------------
    elif model_name == "realesrgan":
        from scripts.models.realesrgan import RealESRGANModule

        return RealESRGANModule.load_from_checkpoint(
            checkpoint,
            **common_norm
        ) if checkpoint else RealESRGANModule(
            pretrained_path=str(PRETRAINED / "RealESRGAN_generator_x4.pth"),
            **common_norm
        )


    # ---------------- ResShift ----------------
    elif model_name == "resshift":
        from scripts.models.resshift import ResShiftModule

        return ResShiftModule.load_from_checkpoint(checkpoint) if checkpoint else ResShiftModule(
            pretrained_path=str(PRETRAINED / "ResShift_x4.pth"),
            ae_path=str(PRETRAINED / "autoencoder_vq_f4.pth"),
            hr_mean=HR_MEAN,
            hr_std=HR_STD,
        )

    raise ValueError(f"Unknown model: {model_name}")


# -----------------------------
# INFERENCE
# -----------------------------
def run_inference(model_name, phase, checkpoint=None,
                  use_water_metrics=False, group=None):

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


# -----------------------------
# BENCHMARK ALL MODELS
# -----------------------------
def run_all(phase, checkpoint_dir=None):

    models = ["edsr", "swinir", "hat", "realesrgan", "resshift"]
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
            f"{metrics.get('test_mae', 0):>8.4f} "
            f"{metrics.get('test_rmse', 0):>8.4f}"
        )

    out = RESULTS_DIR / f"phase{phase}" / f"phase{phase}_comparison.json"
    out.parent.mkdir(parents=True, exist_ok=True)

    with open(out, "w") as f:
        json.dump(all_results, f, indent=2)

    print(f"\nSaved → {out}")


# -----------------------------
# CLI
# -----------------------------
if __name__ == "__main__":

    parser = argparse.ArgumentParser()

    parser.add_argument("--model", type=str,
                        choices=["edsr", "swinir", "hat", "realesrgan", "resshift"])
    parser.add_argument("--phase", type=int, default=1, choices=[1, 2])
    parser.add_argument("--checkpoint", type=str, default=None)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--use_water_metrics", action="store_true")
    parser.add_argument("--run_name", default=None)
    parser.add_argument("--group", default=None)
    parser.add_argument("--project", default="TIR_sisr")

    args = parser.parse_args()

    if args.all:
        run_all(args.phase)

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