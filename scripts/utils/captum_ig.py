import torch
import numpy as np
import matplotlib.pyplot as plt

import captum
from captum.attr import IntegratedGradients
from pathlib import Path
from torch.utils.data import DataLoader

from scripts.models.realesrgan import RealESRGANModule
from scripts.utils.dataset import SRDataset


# ----------------------------
# CONFIG
# ----------------------------
BASE = Path("/share/home/e2406751/Superresolution-TIR")
PATCHES_DIR = BASE / "data/processed/patches"
STATS_PATH = PATCHES_DIR / "stats.json"
CKPT_PATH = BASE / "checkpoints/gan_v5/exp_1782358324_epoch=44_val_water_mae=0.9543.ckpt"
OUT_DIR = BASE / "results"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ----------------------------
# LOAD MODEL
# ----------------------------
def load_model():
    model = RealESRGANModule.load_from_checkpoint(
        CKPT_PATH,
        map_location=DEVICE
    )
    model.eval()
    model.to(DEVICE)
    return model


# ----------------------------
# DATA
# ----------------------------
def get_loader():
    ds = SRDataset(
        split="test",
        patches_dir=PATCHES_DIR,
        stats_path=STATS_PATH,
        use_aux=True,
        use_water_mask=True,
        aux_dir=str(PATCHES_DIR / "test" / "AUX"),
        transform=None
    )
    return DataLoader(ds, batch_size=1, shuffle=False)


# ----------------------------
# WRAPPER (Captum compatible)
# ----------------------------
class WrappedModel(torch.nn.Module):
    def __init__(self, model, aux_chans):
        super().__init__()
        self.model = model
        self.aux_chans = aux_chans

    def forward(self, x, aux_mid, aux_hr):
        lr = x[:, 0:1]
        aux = x[:, 1:1 + self.aux_chans]
        time = x[:, 1 + self.aux_chans:]

        batch = {
            "lr": lr,
            "aux_lr": aux,
            "aux_mid": aux_mid,
            "aux_hr": aux_hr,
            "time_channels": time,
        }

        out = self.model(batch)

        # IMPORTANT: mask-aware scalar
        return out.mean(dim=(1, 2, 3))


# ----------------------------
# BUILD INPUT
# ----------------------------
def build_input(batch):
    return torch.cat([
        batch["lr"],
        batch["aux_lr"],
        batch["time_channels"],
    ], dim=1)


# ----------------------------
# RUN IG
# ----------------------------
def run():
    model = load_model()
    loader = get_loader()

    batch = next(iter(loader))

    batch = {
        k: v.to(DEVICE) if torch.is_tensor(v) else v
        for k, v in batch.items()
    }

    x = build_input(batch).to(DEVICE)
    x.requires_grad_(True)

    wrapped = WrappedModel(model, model.aux_chans).to(DEVICE)
    wrapped.eval()

    ig = IntegratedGradients(wrapped)

    # baseline = "zero input" in normalized space
    baseline = torch.zeros_like(x, device=DEVICE)

    attr = ig.attribute(
                inputs=x,
                baselines=baseline,
                additional_forward_args=(
                    batch["aux_mid"],
                    batch["aux_hr"],
                ),
                n_steps=64,
            )

    # ----------------------------
    # CHANNEL IMPORTANCE
    # ----------------------------
    attr_abs = attr.abs().mean(dim=(0, 2, 3))  # [C]

    lr_score = attr_abs[0].item()
    aux_scores = attr_abs[1:1 + model.aux_chans].detach().cpu().numpy()
    time_scores = attr_abs[1 + model.aux_chans:].detach().cpu().numpy()

    print("\n=== Integrated Gradients Channel Importance ===")
    print(f"LR importance   : {lr_score:.6f}")
    print(f"Aux mean        : {aux_scores.mean():.6f}")
    print(f"Time mean       : {time_scores.mean():.6f}")

    # ----------------------------
    # PLOTS (saved, not shown)
    # ----------------------------

    plt.figure()
    plt.title("Aux Channel Importance (IG)")
    plt.bar(np.arange(len(aux_scores)), aux_scores)
    plt.xlabel("Aux channel")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "aux_importance.png", dpi=200)
    plt.close()

    plt.figure()
    plt.title("Time Channel Importance (IG)")
    plt.bar(np.arange(len(time_scores)), time_scores)
    plt.xlabel("Time channel (0=hour,1=gap_hr,2=gap_day)")
    plt.ylabel("Attribution")
    plt.tight_layout()
    plt.savefig(OUT_DIR / "time_importance.png", dpi=200)
    plt.close()

    print(f"\nSaved results to: {OUT_DIR}")


if __name__ == "__main__":
    run()