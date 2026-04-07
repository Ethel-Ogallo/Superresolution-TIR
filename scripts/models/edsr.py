"""
edsr.py — EDSR Lightning module for TIR super-resolution (x4).

Data flow:
  dataset  →  (1, H, W)  →  repeat(1,3,1,1)  →  EDSR backbone  →  (1, H*4, W*4)

Loss:
  L_total = L1(SR, HR) + lambda_grad * GradientLoss(SR, HR)

  GradientLoss uses kornia.filters.SpatialGradient to compute spatial
  temperature gradients (rate of change between pixels in x and y).
  Only valid (non-nodata) pixels contribute to both terms.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import edsr_arch
from kornia.filters import SpatialGradient
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure



# ----------Gradient loss ---------------
_spatial_gradient = SpatialGradient()  # shared instance, no re-init overhead

def gradient_loss(sr: torch.Tensor, hr: torch.Tensor,
                  mask: torch.Tensor) -> torch.Tensor:
    """
    Spatial temperature gradient loss.
    Uses kornia.filters.SpatialGradient - output shape (B, 1, 2, H, W)
    where dim 2 is (x_gradient, y_gradient).
    Only valid (non-nodata) pixels contribute.

    sr, hr, mask : (B, 1, H, W) — in physical units (°C)
    """
    sr_grads = _spatial_gradient(sr)   # (B, 1, 2, H, W)
    hr_grads = _spatial_gradient(hr)   # (B, 1, 2, H, W)

    # Erode mask by 1px — border pixels produce artificial gradients at nodata edges
    mask_inner = (F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) > 0.99).float()
    mask_5d    = mask_inner.unsqueeze(2).expand_as(sr_grads)   # (B, 1, 2, H, W)

    n    = torch.clamp(mask_5d.sum(), min=1.0)
    loss = (torch.abs(sr_grads - hr_grads) * mask_5d).sum() / n
    return loss


# ------------------- Model ----------------------------

class EDSRModule(pl.LightningModule):

    DATA_RANGE = 60.0   # °C — used by PSNR / SSIM

    def __init__(
        self,
        pretrained_path:   str   = None,
        mean:              float = 0.0,
        std:               float = 1.0,
        learning_rate:     float = 1e-4,
        bb_lr_scale: float = 0.1,
        patience:          int   = 5,
        n_feats:           int   = 64,
        n_blocks:          int   = 16,
        freeze_backbone:   bool  = True,
        lambda_grad:       float = 0.1,   
    ):
        super().__init__()
        self.save_hyperparameters()

        # Backbone: 3-ch in, 1-ch out 
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
            print("[INFO] No pretrained weights — training EDSR from scratch")

        # Reset final conv to 1-ch output for TIR
        self.body.conv_last = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self.body.mean      = torch.zeros(1, 1, 1, 1)

        self._set_backbone_frozen(freeze_backbone)

        # Metrics
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ------- Helpers -------------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        """Crop to tight bounding box of valid (mask=1) pixels."""
        m    = mask[:, 0, :, :]
        flat = m.any(dim=0)
        rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
        cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x : (B, 1, H, W) → sr : (B, 1, H*4, W*4)"""
        x = x.repeat(1, 3, 1, 1)
        return self.body(x)

    # -------------------- Shared step --------------------
    def _shared_step(self, batch, stage: str):
        lr_img, hr_img, hr_mask = batch   # all (B, 1, H, W)
        sr_img = self(lr_img)             # (B, 1, H*4, W*4)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        # Reconstruction loss (masked L1) 
        abs_err   = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        # Gradient loss (masked Sobel) 
        if self.hparams.lambda_grad > 0:
            grad_loss = gradient_loss(sr, hr, hr_mask)
            loss      = recon_loss + self.hparams.lambda_grad * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=sr.device)
            loss      = recon_loss

        # Metrics — crop to valid bbox to avoid nodata skewing PSNR/SSIM
        with torch.no_grad():
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _         = self.crop_to_valid_bbox(hr, hr_mask)
            psnr_val = getattr(self, f"{stage}_psnr")(sr_crop * mask_crop,
                                                       hr_crop * mask_crop)
            ssim_val = getattr(self, f"{stage}_ssim")(sr_crop, hr_crop)

        self.log(f"{stage}_loss",      loss,      on_epoch=True, prog_bar=True)
        self.log(f"{stage}_recon_loss",recon_loss, on_epoch=True, prog_bar=False)
        self.log(f"{stage}_grad_loss", grad_loss,  on_epoch=True, prog_bar=False)
        self.log(f"{stage}_psnr",      psnr_val,  on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim",      ssim_val,  on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ------------------- Optimizer --------------------
    def configure_optimizers(self):
        backbone = []
        head = []

        for name, p in self.body.named_parameters():
            if not p.requires_grad:
                continue
            if "conv_last" in name:
                head.append(p)
            else:
                backbone.append(p)

        optimizer = optim.Adam([
            {"params": backbone, "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
            {"params": head,     "lr": self.hparams.learning_rate},
        ], weight_decay=1e-6)

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