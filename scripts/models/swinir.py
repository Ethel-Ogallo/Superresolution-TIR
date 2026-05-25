# scripts/models/swinir.py
"""
SwinIR Lightning module for TIR Super-Resolution.
Phase 1: zero-shot inference
Phase 2: fine-tuning with masked L1
Architecture untouched — 3ch in, 3ch out, metrics on ch0 only.
"""

import os
import torch
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str  = None,
        hr_mean: float        = 0.0,
        hr_std: float         = 1.0,
        learning_rate: float  = 1e-4,
        img_size: int         = 48,
        embed_dim: int        = 180,
        depths: list          = None,
        num_heads: list       = None,
        window_size: int      = 8,
        mlp_ratio: float      = 2.0,
        upsampler: str        = "pixelshuffle",
        data_range: float     = None,
        data_min: float       = None,
        phase: int            = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_RANGE = float(data_range)
        self.DATA_MIN   = float(data_min)

        depths    = depths    or [6, 6, 6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6, 6, 6]

        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=3,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection="1conv",
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
                "lr_scheduler": {"scheduler": sch, "monitor": "val_full_psnr"}}

    def _load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Pretrained not found: {path}")
            return
        print(f"[INFO] Loading SwinIR pretrained: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)
        model_dict = self.body.state_dict()
        matched    = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} layers")