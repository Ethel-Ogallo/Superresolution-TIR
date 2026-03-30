"""
edsr.py — EDSR Lightning module for TIR super-resolution (×4).

Fully self-contained.

Data flow:
  dataset  →  (1, H, W)  →  repeat(1,3,1,1)  →  EDSR backbone  →  (1, H*4, W*4)

Architecture changes vs. RGB baseline:
  - Input  : 1-channel TIR repeated to 3 channels inside forward()
  - Output : 1-channel temperature map (conv_last reset to out_ch=1)
  - Pretrained weights loaded with key remapping; tail/mean layers skipped
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import edsr_arch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class EDSRModule(pl.LightningModule):

    DATA_RANGE = 60.0   # °C — used by PSNR / SSIM

    def __init__(
        self,
        pretrained_path:   str   = None,
        mean:              float = 0.0,
        std:               float = 1.0,
        learning_rate:     float = 1e-4,
        backbone_lr_scale: float = 0.1,
        patience:          int   = 5,
        n_feats:           int   = 64,
        n_blocks:          int   = 16,
        freeze_backbone:   bool  = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ------- Backbone: 3-ch in, 1-ch out --------
        self.body = edsr_arch.EDSR(
            num_in_ch  = 3,
            num_out_ch = 3,     
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
            print("[INFO] No pretrained weights — training EDSR from scratch")

        # Reset final conv to 1-ch output for TIR
        self.body.conv_last = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self.body.mean      = torch.zeros(1, 1, 1, 1)

        self._set_backbone_frozen(freeze_backbone)

        # ------------ Metrics -----------
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ------------- Helpers ----------------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        """Works for any (B, C, H, W) tensor regardless of C."""
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        """Crop to tight bounding box of valid (mask=1) pixels. Works for C=1."""
        # mask: (B, 1, H, W) — use channel 0 to find spatial bbox
        m    = mask[:, 0, :, :]          # (B, H, W)
        flat = m.any(dim=0)              # (H, W) — any batch item valid
        rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
        cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

    # ------------- Pretrained loading ----------------
    def _load_pretrained(self, path: str):
        self.print(f"[INFO] Loading pretrained weights: {path}")
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
        self.print(f"[INFO] Matched {len(matched)} / {len(model_dict)} layers")

    # ---------- Freeze logic -------------------
    def _set_backbone_frozen(self, frozen: bool):
        for name, param in self.body.named_parameters():
            param.requires_grad = (
                (not frozen)
                or ("conv_last" in name)
                or ("upsample"  in name)
            )

    # ---------- Forward -------------------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, H, W) → sr : (B, 1, H*4, W*4)"""
        x = x.repeat(1, 3, 1, 1)   # (B,1,H,W) → (B,3,H,W)
        return self.body(x)         # → (B,1,H*4,W*4)

    # ---------- Shared step --------------
    def _shared_step(self, batch, stage: str):
        lr_img, hr_img, hr_mask = batch   # all (B, 1, H, W)
        sr_img = self(lr_img)             # (B, 1, H*4, W*4)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        # Masked L1 — only valid HR pixels contribute
        abs_err = torch.abs(sr - hr) * hr_mask
        loss    = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        with torch.no_grad():
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _         = self.crop_to_valid_bbox(hr, hr_mask)
            psnr_val = getattr(self, f"{stage}_psnr")(sr_crop * mask_crop,
                                                       hr_crop * mask_crop)
            ssim_val = getattr(self, f"{stage}_ssim")(sr_crop, hr_crop)

        self.log(f"{stage}_loss", loss,     on_epoch=True, prog_bar=True)
        self.log(f"{stage}_psnr", psnr_val, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim", ssim_val, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ------------- Optimizer ------------
    def configure_optimizers(self):
        backbone_params = [p for p in self.body.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            [{"params": backbone_params,
              "lr": self.hparams.learning_rate * self.hparams.backbone_lr_scale}],
            weight_decay=1e-6,
        )
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max", 
            factor=0.5, 
            patience=self.hparams.patience,
        )
        return {
            "optimizer":    optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_psnr"},
        }