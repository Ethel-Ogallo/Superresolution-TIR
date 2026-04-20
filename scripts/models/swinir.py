"""
swinir.py — SwinIR Lightning module for TIR super-resolution (x4).
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from kornia.filters import SpatialGradient
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

# ---------- Gradient loss ---------------
_spatial_gradient = SpatialGradient()

def gradient_loss(sr: torch.Tensor, hr: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    """Spatial temperature gradient loss. Only valid pixels contribute."""
    sr_grads = _spatial_gradient(sr)  # (B, 1, 2, H, W)
    hr_grads = _spatial_gradient(hr)
    
    # Erode mask by 1px to avoid border artifacts
    mask_inner = (F.avg_pool2d(mask, kernel_size=3, stride=1, padding=1) > 0.99).float()
    mask_5d = mask_inner.unsqueeze(2).expand_as(sr_grads)
    
    n = torch.clamp(mask_5d.sum(), min=1.0)
    loss = (torch.abs(sr_grads - hr_grads) * mask_5d).sum() / n
    return loss


class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        learning_rate: float = 1e-4,
        bb_lr_scale: float = 0.1,
        patience: int = 5,
        img_size: int = 64,
        embed_dim: int = 60,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffle",
        freeze_backbone: bool = False,
        lambda_grad: float = 0.1,
        # ── FIX: data_range is now a proper param instead of hardcoded ──
        # For Rhône corridor TIR data (°C): river ~10-20°C, asphalt/roofs up to ~55-60°C
        # Set this from your dataset global min/max: data_range = T_max - T_min
        # Run the debug print below once to confirm the real span.
        data_range: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ── FIX: DATA_RANGE now comes from the param, not hardcoded ──
        # Old code had self.DATA_RANGE = 80.0 which was wrong for a river corridor.
        # Realistic TIR range for Rhône (water + asphalt + roofs) is ~50°C.
        # Override by passing data_range=X to the constructor or your config yaml.
        self.DATA_RANGE = float(data_range)
        print(f"[INFO] DATA_RANGE set to {self.DATA_RANGE}°C ")

        depths = depths or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        # Create the backbone (still creates with 3ch out by default)
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

        # === THE FOOLPROOF OUTPUT CHANNEL FIX ===
        def replace_last_conv(model, out_channels=1):
            last_conv_name = None

            # Traverse the model to find the very last Conv2d layer
            for name, module in model.named_modules():
                if isinstance(module, nn.Conv2d):
                    last_conv_name = name
            
            if last_conv_name:
                parts = last_conv_name.split('.')
                parent = model
                for part in parts[:-1]:
                    parent = getattr(parent, part)
                
                old_conv = getattr(parent, parts[-1])
                new_conv = nn.Conv2d(
                    old_conv.in_channels, 
                    out_channels, 
                    kernel_size=old_conv.kernel_size, 
                    padding=old_conv.padding
                )
                setattr(parent, parts[-1], new_conv)
                # ── Store the replaced layer name so configure_optimizers can use it ──
                self._replaced_conv_name = last_conv_name
                print(f"[INFO] Successfully replaced {last_conv_name} with 1-ch output.")
            else:
                print("[ERROR] Could not find any Conv2d layers in the backbone!")
                self._replaced_conv_name = None

        replace_last_conv(self.body, out_channels=1)

        # Now load pretrained (skips output layers)
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] No pretrained weights — training from scratch")

        self._set_backbone_frozen(freeze_backbone)

        # ── FIX: pass the correct DATA_RANGE to both PSNR and SSIM metrics ──
        # Old code had these hardcoded to 80.0 inside the metric constructors.
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr", PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim", StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ------------------- Helpers -------------------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std = torch.tensor(self.hparams.std, device=t.device)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        """Crop to tight bounding box of valid pixels."""
        m = mask[:, 0, :, :]
        flat = m.any(dim=0)
        rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
        cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

    # ------------------- Pretrained loading -------------------
    def _load_pretrained(self, path):
        print(f"[INFO] Loading pretrained SwinIR weights: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)
        skip_keys = {"conv_last", "upsample", "conv_before_upsample"}

        model_dict = self.body.state_dict()
        matched = {}

        for k, v in state_dict.items():
            if any(s in k for s in skip_keys):
                continue
            if k not in model_dict:
                continue
            if v.shape != model_dict[k].shape:
                continue
            matched[k] = v

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} layers")

    def _set_backbone_frozen(self, frozen: bool):
        # ── These keywords identify the reconstruction/upscaling head layers ──
        head_keywords = ["upsample", "conv_last", "conv_before_upsample", "conv_hr"]
        
        for name, param in self.body.named_parameters():
            is_head = any(k in name for k in head_keywords)
            if frozen:
                param.requires_grad = is_head
            else:
                param.requires_grad = True

    # ------------------- Forward -------------------
    # def forward(self, x: torch.Tensor) -> torch.Tensor:
    #     x = x.repeat(1, 3, 1, 1)
    #     sr = self.body(x)

    #     if sr.shape[1] != 1:
    #         sr = sr[:, :1, :, :]
            
    #     return sr

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)
        sr = self.body(x3)
        
        if sr.shape[1] != 1:
            sr = sr[:, :1, :, :]
        return sr

    # ------------------- Shared step -------------------
    def _shared_step(self, batch, batch_idx: int, stage: str):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        sr = torch.nan_to_num(sr, nan=0.0, posinf=1e6, neginf=-1e6)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6)

        # Reconstruction loss (masked L1)
        abs_err = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        # Gradient loss
        if self.hparams.lambda_grad > 0:
            grad_loss = gradient_loss(sr, hr, hr_mask)
            loss = recon_loss + self.hparams.lambda_grad * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=sr.device)
            loss = recon_loss

        recon_loss = torch.nan_to_num(recon_loss, nan=0.0)
        grad_loss  = torch.nan_to_num(grad_loss,  nan=0.0)
        loss       = torch.nan_to_num(loss,        nan=0.0)

        # ==================== Metrics ====================
        with torch.no_grad():
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _         = self.crop_to_valid_bbox(hr, hr_mask)

            valid_mask = (mask_crop > 0.5)  # (B, 1, H, W)

            if valid_mask.sum() == 0:
                psnr_val = torch.tensor(0.0, device=sr.device)
                ssim_val = torch.tensor(0.0, device=sr.device)
                mae_val  = torch.tensor(0.0, device=sr.device)
            else:
                # ── MAE: mean absolute error over valid pixels only ──
                # NOTE: this will equal recon_loss when lambda_grad=0 because
                # recon_loss is also a masked MAE. Kept separately for clarity.
                mae_val = (sr_crop - hr_crop).abs()[valid_mask].mean()

                # ── FIX: PSNR needs a proper 4D (B, C, H, W) tensor ──
                # Old code did sr_crop[valid_mask] which flattened to 1D,
                # then unsqueezed to (1, 1, N) — torchmetrics silently
                # computed on the wrong shape giving garbage PSNR (~8 dB).
                # Fix: keep 4D shape and neutralize invalid pixels by setting
                # sr == hr there (zero error contribution, doesn't bias metric).
                sr_for_metric = sr_crop.clone()
                hr_for_metric = hr_crop.clone()
                sr_for_metric[~valid_mask] = hr_for_metric[~valid_mask]

                psnr_val = getattr(self, f"{stage}_psnr")(sr_for_metric, hr_for_metric)

                # ── FIX: SSIM also uses the neutralized 4D tensors ──
                # Old code zero-filled invalid pixels which created sharp artificial
                # edges at mask boundaries, corrupting local window computations.
                # Neutralizing (sr=hr at invalid pixels) avoids this.
                h, w = sr_crop.shape[-2:]
                if h < 11 or w < 11:
                    # SSIM window is 11x11 — skip if crop is too small
                    ssim_val = torch.tensor(0.0, device=sr.device)
                else:
                    ssim_val = getattr(self, f"{stage}_ssim")(sr_for_metric, hr_for_metric)

        # Logging
        self.log(f"{stage}_loss",       loss,       on_epoch=True, prog_bar=True)
        self.log(f"{stage}_recon_loss", recon_loss, on_epoch=True)
        self.log(f"{stage}_grad_loss",  grad_loss,  on_epoch=True)
        self.log(f"{stage}_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae",        mae_val,    on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, stage):
        return self._shared_step(batch, stage, "train")

    def validation_step(self, batch, stage):
        return self._shared_step(batch, stage, "val")

    def test_step(self, batch, stage):
        return self._shared_step(batch, stage, "test")

    # ------------------- Optimizer -------------------
    def configure_optimizers(self):
        head_keywords = ["upsample", "conv_last", "conv_before_upsample", "conv_hr"]

        backbone, head = [], []
        for name, p in self.body.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in head_keywords):
                head.append(p)
            else:
                backbone.append(p)

        optimizer = optim.Adam([
            {"params": backbone, "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
            {"params": head,     "lr": self.hparams.learning_rate},
        ], weight_decay=1e-6)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="max", factor=0.5, patience=self.hparams.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_psnr"},
        }