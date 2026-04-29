"""
edsr.py — EDSR Lightning module for TIR super-resolution (x4).
- Backbone adapted from BasicSR's EDSR implementation, with option to load pretrained weights.
- Supports optional AUX encoder branch for additional HR inputs.
- Training/validation/test steps compute masked L1 loss + gradient loss, and log PSNR/SSIM metrics.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import edsr_arch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.aux_encoder import AuxEncoder
from scripts.utils.metrics import shared_step


class EDSRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path:   str   = None,
        mean:              float = 0.0,
        std:               float = 1.0,
        learning_rate:     float = 1e-4,
        bb_lr_scale:       float = 0.1,
        patience:          int   = 5,
        n_feats:           int   = 64,
        n_blocks:          int   = 16,
        freeze_backbone:   bool  = True,
        lambda_grad:       float = 0.1,
        data_range:        float = 70.0,
        use_aux:           bool  = False, 
        aux_channels:      int   = 10,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_RANGE = float(data_range)
        self.use_aux    = use_aux
        print(f"[INFO] DATA_RANGE set to {self.DATA_RANGE}°C")

        # Backbone — 3ch in for pretrained weights, 1ch out for TIR
        self.body = edsr_arch.EDSR(
            num_in_ch  = 3,
            num_out_ch = 1,
            num_feat   = n_feats,
            num_block  = n_blocks,
            upscale    = 4,
            res_scale  = 0.1,
            img_range  = 1.0,
            rgb_mean   = (0.0, 0.0, 0.0),
        )

        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] No pretrained weights; training EDSR from scratch")

        # Reset final conv to 1ch output for TIR
        self.body.conv_last = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self.body.mean      = torch.zeros(1, 1, 1, 1)

        self._set_backbone_frozen(freeze_backbone)

        # AUX encoder — separate branch, does not touch backbone
        if use_aux:
            self.aux_encoder = AuxEncoder(
                in_channels=aux_channels,
                base_channels=n_feats
            )
            # fuse upsampled LR features + AUX features at HR resolution
            self.fusion = nn.Conv2d(n_feats * 2, n_feats, kernel_size=1)
            print(f"[INFO] AUX encoder enabled with {aux_channels} input channels")
        else:
            print("[INFO] AUX encoder disabled; standard EDSR")

        # Metrics
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr", PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim", StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ------- Helpers -------------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    # --------------- Pretrained loading -------
    def _load_pretrained(self, path: str):
        print(f"[INFO] Loading pretrained weights: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)

        def remap_key(k: str):
            if k.startswith(("tail", "sub_mean", "add_mean")):
                return None
            if k.startswith("head.0."):
                return k.replace("head.0.", "conv_first.", 1)
            if k.startswith("body."):
                if k.startswith("body.16."):
                    return k.replace("body.16.", "conv_after_body.", 1)
                k = k.replace(".body.0.", ".conv1.", 1)
                k = k.replace(".body.2.", ".conv2.", 1)
                return k
            return None

        model_dict = self.body.state_dict()
        matched    = {}
        for ckpt_key, v in state_dict.items():
            model_key = remap_key(ckpt_key)
            if model_key is None or model_key not in model_dict:
                continue
            if v.shape != model_dict[model_key].shape:
                continue
            matched[model_key] = v

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Matched {len(matched)} / {len(model_dict)} layers")

    # Freeze logic
    def _set_backbone_frozen(self, frozen: bool):
        for name, param in self.body.named_parameters():
            param.requires_grad = (
                (not frozen)
                or ("conv_last" in name)
                or ("upsample"  in name)
            )

    # Forward
    def forward(self, lr: torch.Tensor, aux: torch.Tensor = None):

        lr_3ch = lr.repeat(1, 3, 1, 1) #repeat 1ch TIR to 3ch for EDSR backbone, since pretrained weights expect 3 channels

        #  EDSR backbone 
        lr_feats = self.body.conv_first(lr_3ch)
        res = self.body.body(lr_feats)
        res = self.body.conv_after_body(res) + lr_feats

        # AUX branch
        if self.use_aux and aux is not None:
            aux_feats = self.aux_encoder(aux)
            # align feature spaces BEFORE upsample fusion
            fused = torch.cat([res, aux_feats], dim=1)
            fused = self.fusion(fused)            
            out = self.body.upsample(fused) # NOW upsample once after fusion

            return self.body.conv_last(out)

        # no AUX, standard EDSR flow
        out = self.body.upsample(res)
        return self.body.conv_last(out)

    # -------------------- Steps --------------------
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # ------------------- Optimizer --------------------
    def configure_optimizers(self):
        backbone   = []
        head       = []
        aux_params = []

        for name, p in self.body.named_parameters():
            if not p.requires_grad:
                continue
            if "conv_last" in name:
                head.append(p)
            else:
                backbone.append(p)

        # AUX encoder trains at full LR
        if self.use_aux:
            aux_params = (
                list(self.aux_encoder.parameters()) +
                list(self.fusion.parameters())
            )

        param_groups = [
            {"params": backbone,   "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
            {"params": head,       "lr": self.hparams.learning_rate},
        ]
        if aux_params:
            param_groups.append(
                {"params": aux_params, "lr": self.hparams.learning_rate}
            )

        optimizer = optim.Adam(param_groups, weight_decay=1e-6)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self.hparams.patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_psnr"},
        }