"""
evaluate.py — Model-agnostic evaluation for TIR SR models.

Saves metrics and optional LR/SR/HR outputs to W&B artifacts.
Minimal local disk usage (cluster-friendly).
"""

import argparse
import os
import torch
from torch.utils.data import DataLoader
import pandas as pd
import wandb

from scripts.utils.dataset import SRDataset

# ------------- Model registry -----------
def load_model(ckpt_path: str, device: torch.device):
    from scripts.models.edsr import EDSRModule
    REGISTRY = {"EDSRModule": EDSRModule}

    if ckpt_path.startswith("wandb:"):
        api   = wandb.Api()
        art   = api.artifact(ckpt_path.removeprefix("wandb:"))
        local = art.download()
        ckpt_path = next(
            os.path.join(local, f) for f in os.listdir(local) if f.endswith(".ckpt")
        )

    raw      = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    hyper    = raw.get("hyper_parameters", {})
    cls_name = hyper.get("model_class", None)

    if cls_name in REGISTRY:
        model = REGISTRY[cls_name].load_from_checkpoint(ckpt_path, map_location=device)
        return model.eval().to(device), os.path.basename(ckpt_path)

    # fallback scan
    for cls in REGISTRY.values():
        try:
            model = cls.load_from_checkpoint(ckpt_path, map_location=device)
            return model.eval().to(device), os.path.basename(ckpt_path)
        except Exception:
            continue
    raise ValueError("[INFO] Could not infer model class from checkpoint.")


# ------------- Main  ------------------
def main():
    args   = parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    os.chdir(os.path.dirname(os.path.abspath(__file__)) + "/../..")

    # Start W&B run
    run_name = args.run_name or f"eval_{args.split}"
    wandb.init(project=args.project, name=run_name)

    # Load metadata
    metadata = pd.read_csv(args.metadata_csv)
    subset   = metadata[metadata["split"] == args.split].reset_index(drop=True)
    print(f"[INFO] Evaluating {len(subset)} patches from '{args.split}' split.")

    # Load model
    model, ckpt_name = load_model(args.checkpoint, device)
    mean, std       = model.hparams.mean, model.hparams.std

    # Dataset + loader
    ds     = SRDataset(subset, mean=mean, std=std)
    loader = DataLoader(ds, batch_size=1, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True)

    # Evaluate metrics
    from lightning.pytorch import Trainer
    trainer = Trainer(accelerator="auto", devices="auto", logger=False)
    results = trainer.test(model, dataloaders=loader)

    # Log metrics to W&B table
    table = wandb.Table(columns=["patch", *results[0].keys()])
    for i, row in enumerate(results):
        table.add_data(i, *[v for v in row.values()])
    wandb.log({"metrics_table": table})

    # Optional outputs
    if args.save_outputs:
        outputs_art = wandb.Artifact(f"{ckpt_name}_outputs", type="dataset")
        n_save = min(args.n_vis, len(ds))
        for i in range(n_save):
            sample = ds[i]
            lr, hr = sample[:2]
            with torch.no_grad():
                sr = model(lr.unsqueeze(0).to(device))
            # Here you would serialize tensors to W&B artifact (no local storage)
            # Example placeholder: outputs_art.add_file("temp_sr_{i:03d}.pt")
        wandb.log_artifact(outputs_art)

    wandb.finish()
    print(f"[INFO] Evaluation done. Metrics logged to W&B run '{wandb.run.name}'.")

# ------------- CLI  -------------
def parse_args():
    p = argparse.ArgumentParser(description="Evaluate any TIR SR checkpoint")
    p.add_argument("--checkpoint", required=True,
                   help="Path to .ckpt or 'wandb:entity/project/model-ID:best'")
    p.add_argument("--metadata_csv", default="metadata.csv")
    p.add_argument("--split", choices=["train", "val", "test"], default="test")
    p.add_argument("--num_workers", type=int, default=2)
    p.add_argument("--save_outputs", action="store_true", help="Save LR/SR/HR tensors to W&B artifact")
    p.add_argument("--n_vis", type=int, default=5)
    p.add_argument("--project", default="TIR_sisr")
    p.add_argument("--run_name", default=None)
    return p.parse_args()


if __name__ == "__main__":
    main()