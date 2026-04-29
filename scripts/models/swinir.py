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
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.loss import gradient_loss

class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        learning_rate: float = 1e-4,
        bb_lr_scale: float = 0.1,
        patience: int = 5,
        img_size: int = 48,
        embed_dim: int = 180,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 8,
        mlp_ratio: float = 2.0,
        upsampler: str = "pixelshuffle",
        freeze_backbone: bool = False,
        lambda_grad: float = 0.1,
        data_range: float = 80.0, 
    ):
        super().__init__()
        self.save_hyperparameters()

        self.DATA_RANGE = float(data_range)
        print(f"[INFO] DATA_RANGE set to {self.DATA_RANGE}°C ")

        depths = depths or [6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6]

        # Create the backbone SwinIR model with the specified config
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

        # Output channel
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
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)
        sr = self.body(x3)
        
        if sr.shape[1] != 1:
            sr = sr[:, :1, :, :]
        return sr

    # ------------------- Shared step -------------------
    def _shared_step(self, batch, stage: str):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        # >>>>> NEW START
        # 1. Check if we have any valid data in this batch
        mask_sum = hr_mask.sum()
        if mask_sum < 1.0:
            # Multiply the model output by 0.0
            # This 'chains' the model to the loss so the scaler stays happy,
            # but the actual gradient value will be 0, so no weights change.
            return sr_img.sum() * 0.0

        # 2. Reconstruction loss (masked L1)
        abs_err = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / mask_sum

        # 3. Gradient loss (using your existing function)
        if self.hparams.lambda_grad > 0:
            grad_loss = gradient_loss(sr, hr, hr_mask)
            loss = recon_loss + self.hparams.lambda_grad * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=sr.device)
            loss = recon_loss
        # >>>NEW END

        # METRICS 
        with torch.no_grad():
            # 1. Focus only on the area with the flight strip
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _ = self.crop_to_valid_bbox(hr, hr_mask)

            valid = (mask_crop > 0.5)

            if valid.sum() == 0:
                psnr_val = ssim_val = torch.tensor(0.0, device=sr.device)
            else:
                # Prepare images for metrics
                # We zero out everything outside the mask for BOTH images.
                # This makes the backgrounds identical so SSIM ignores them.
                sr_clean = sr_crop * mask_crop
                hr_clean = hr_crop * mask_crop

                #  PSNR Calculation
                psnr_val = getattr(self, f"{stage}_psnr")(sr_clean, hr_clean)

                # SSIM Calculation
                h, w = sr_crop.shape[-2:]
                if min(h, w) < 11:
                    ssim_val = torch.tensor(0.0, device=sr.device)
                else:
                    # By having 0 in the background of both, SSIM focuses 
                    # only on the structural difference of the river strip.
                    ssim_val = getattr(self, f"{stage}_ssim")(sr_clean, hr_clean)

            if valid.sum() == 0:
                mae_val = torch.tensor(0.0, device=sr.device)
            else:
                err = sr_crop - hr_crop
                err = err[valid]

                mae_val = err.abs().mean()

        # Logging
        self.log(f"{stage}_loss",       loss,       on_epoch=True, prog_bar=True)
        self.log(f"{stage}_recon_loss", recon_loss, on_epoch=True)
        self.log(f"{stage}_grad_loss",  grad_loss,  on_epoch=True)
        self.log(f"{stage}_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae",        mae_val,    on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

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