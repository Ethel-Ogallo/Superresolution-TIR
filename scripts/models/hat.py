# scripts/models/hat.py
"""
HAT Lightning module for TIR Super-Resolution.
Phase 1: zero-shot inference
Phase 2: fine-tuning with masked L1
Architecture untouched — 3ch in, 3ch out, metrics on ch0 only.
"""

import os
import sys
import types
import torch
import torch.optim as optim
import lightning.pytorch as pl
from scripts.utils.metrics import shared_step


def _load_hat_arch():
    try:
        from basicsr.archs import hat_arch
        print("[INFO] HAT arch loaded from basicsr")
        return hat_arch
    except ImportError:
        pass
    if "hat_arch" in sys.modules:
        return sys.modules["hat_arch"]
    import requests
    print("[INFO] Fetching HAT arch from GitHub...")
    url  = "https://raw.githubusercontent.com/XPixelGroup/HAT/main/hat/archs/hat_arch.py"
    resp = requests.get(url, timeout=30)
    resp.raise_for_status()
    mod  = types.ModuleType("hat_arch")
    exec(resp.text, mod.__dict__)
    sys.modules["hat_arch"] = mod
    return mod


_hat_arch = _load_hat_arch()
HAT       = _hat_arch.HAT


class HATModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str  = None,
        hr_mean: float        = 0.0,
        hr_std: float         = 1.0,
        learning_rate: float  = 1e-4,
        img_size: int         = 64,
        embed_dim: int        = 180,
        depths: list          = None,
        num_heads: list       = None,
        window_size: int      = 16,
        mlp_ratio: float      = 2.0,
        compress_ratio: int   = 3,
        squeeze_factor: int   = 30,
        overlap_ratio: float  = 0.5,
        upsampler: str        = "pixelshuffle",
        data_range: float     = None,
        phase: int            = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_RANGE = float(data_range)

        depths    = depths    or [6, 6, 6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6, 6, 6]

        self.body = HAT(
            upscale        = 4,
            in_chans       = 3,
            img_size       = img_size,
            window_size    = window_size,
            compress_ratio = compress_ratio,
            squeeze_factor = squeeze_factor,
            conv_scale     = 0.01,
            overlap_ratio  = overlap_ratio,
            img_range      = 1.0,
            depths         = depths,
            embed_dim      = embed_dim,
            num_heads      = num_heads,
            mlp_ratio      = mlp_ratio,
            upsampler      = upsampler,
            resi_connection= "1conv",
        )

        if pretrained_path:
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] No pretrained path — random init")

    def forward(self, lr):
        return self.body(lr)           # (B, 3, 256, 256)

    def denormalize(self, t):
        mean = torch.tensor(self.hparams.hr_mean, device=t.device)
        std  = torch.tensor(self.hparams.hr_std,  device=t.device)
        return t * std + mean

    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-6
        )
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val_psnr"}}

    def _load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Pretrained not found: {path}")
            return
        print(f"[INFO] Loading HAT pretrained: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
        model_dict = self.body.state_dict()
        matched    = {
            k: v for k, v in state_dict.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
        }
        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} layers")