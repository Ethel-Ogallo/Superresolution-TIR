"""
realesrgan.py — RealESRGAN Lightning module for TIR super-resolution (x4).

Architecture:
  Generator   : RRDBNet (basicsr.archs.rrdbnet_arch) — the proper RealESRGAN
                backbone, NOT SRVGGNetCompact which is the anime/v2 variant.
  Discriminator: UNetDiscriminatorSN (basicsr.archs.discriminator_arch)

Loss surface:
  L1 reconstruction (masked)      — always on
  Spatial gradient loss            — lambda_grad      (mirrors swinir.py)
  Perceptual / VGG feature loss   — lambda_perceptual (basicsr PerceptualLoss)
  Adversarial loss                 — lambda_adversarial

Logging surface (identical to swinir.py so train.py needs no changes):
  {stage}_loss, {stage}_recon_loss, {stage}_grad_loss,
  {stage}_psnr, {stage}_ssim, {stage}_mae
  train_loss_g, train_loss_d, train_perceptual_loss, train_gan_loss_g

Freeze / unfreeze:
  freeze_backbone=True → only the upsampling head of the generator is trained.
  Two param groups in opt_g: backbone (bb_lr_scale × lr) and head (lr).
  Discriminator always trains at d_lr_scale × lr.
"""

import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl

from basicsr.archs import rrdbnet_arch, discriminator_arch
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss
from kornia.filters import SpatialGradient
from torchmetrics.image import (
    PeakSignalNoiseRatio,
    StructuralSimilarityIndexMeasure,
)
from scripts.utils.loss import gradient_loss

# --------------------------------------------------------------------------- #
#  RealESRGANModule                                                            #
# --------------------------------------------------------------------------- #
class RealESRGANModule(pl.LightningModule):
    _HEAD_KEYWORDS = [
        "conv_last", "conv_up1", "conv_up2",
        "conv_hr", "conv_before_upsample",
    ]

    def __init__(
        self,
        pretrained_g_path: str = None,
        pretrained_d_path: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        learning_rate: float = 1e-4,
        bb_lr_scale: float = 0.1,
        d_lr_scale: float = 1.0,
        patience: int = 15,
        n_feats: int = 64,
        n_blocks: int = 23,
        scale: int = 4,
        freeze_backbone: bool = True,
        lambda_grad: float = 0.1,
        lambda_perceptual: float = 1.0,
        lambda_adversarial: float = 0.1,
        data_range: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()
        # GAN requires manual optimisation
        self.automatic_optimization = False

        self.DATA_RANGE = float(data_range)
        print(f"[INFO] DATA_RANGE set to {self.DATA_RANGE}°C")

        # -------------Generator — RRDBNet --------------------
        # in_nc=3 because we repeat the 1-ch TIR image to 3ch before forwarding
        # out_nc=1 — we patch the last conv after loading pretrained weights
        self.net_g = rrdbnet_arch.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,       # patched to 1 after pretrained load
            num_feat=n_feats,
            num_block=n_blocks,
            num_grow_ch=32,
            scale=scale,
        )

        # ----------- Discriminator — UNet with spectral norm -------------
        # Operates on 1-ch TIR images — we wrap to 3ch inside training_step
        self.net_d = discriminator_arch.UNetDiscriminatorSN(
            num_in_ch=1,        # TIR is single-channel
            num_feat=64,
            skip_connection=True,
        )

        #  Load pretrained weights 
        if pretrained_g_path and os.path.exists(pretrained_g_path):
            self._load_generator(pretrained_g_path)
        else:
            print("[INFO] No pretrained generator — training from scratch")

        if pretrained_d_path and os.path.exists(pretrained_d_path):
            self._load_discriminator(pretrained_d_path)
        else:
            print("[INFO] No pretrained discriminator — training from scratch")

        #  Patch generator output to 1-channel AFTER pretrained load 
        self._replaced_conv_name = self._replace_last_conv(out_channels=1)

      
        self._set_backbone_frozen(freeze_backbone)

        #  Losses 
        self.gan_loss = GANLoss(
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=lambda_adversarial,
        )

        # PerceptualLoss: VGG feature matching on 3-ch inputs
        # layer_weights from official RealESRGAN config
        if lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(
                layer_weights={
                    "conv1_2": 0.1,
                    "conv2_2": 0.1,
                    "conv3_4": 1.0,
                    "conv4_4": 1.0,
                    "conv5_4": 1.0,
                },
                vgg_type="vgg19",
                use_input_norm=True,
                range_norm=False,
                perceptual_weight=lambda_perceptual,
                style_weight=0.0,
                criterion="l1",
            )
            for p in self.perceptual_loss.parameters():  
                p.requires_grad = False
        else:
            self.perceptual_loss = None

        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ------------------ Helpers -----------------                                                          
    def _replace_last_conv(self, out_channels: int = 1) -> str | None:
        """Replace the last Conv2d in net_g with a 1-ch output conv."""
        last_conv_name = None
        for name, module in self.net_g.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name

        if last_conv_name is None:
            print("[ERROR] Could not find any Conv2d in net_g!")
            return None

        parts  = last_conv_name.split(".")
        parent = self.net_g
        for part in parts[:-1]:
            parent = getattr(parent, part)

        old_conv = getattr(parent, parts[-1])
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
        )
        setattr(parent, parts[-1], new_conv)
        print(
            f"[INFO] Replaced {last_conv_name}: "
            f"{old_conv.out_channels}ch → {out_channels}ch"
        )
        return last_conv_name

    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        """Crop to the tight bounding box of valid (non-zero) mask pixels."""
        m    = mask[:, 0, :, :]
        flat = m.any(dim=0)
        rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
        cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

    # ---------------- Pretrained loading  --------------------                                             
    def _load_generator(self, path: str):
        print(f"[INFO] Loading pretrained RRDBNet generator: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))

        skip_keys  = {
            "conv_last", "conv_up1", "conv_up2",
            "conv_hr", "conv_before_upsample",
        }
        model_dict = self.net_g.state_dict()

        matched = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
            and not any(s in k for s in skip_keys)
        }

        self.net_g.load_state_dict(matched, strict=False)
        print(f"[INFO] Generator: loaded {len(matched)}/{len(model_dict)} layers")

    def _load_discriminator(self, path: str):
        print(f"[INFO] Loading pretrained discriminator: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params", ckpt)
        model_dict = self.net_d.state_dict()

        matched = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        self.net_d.load_state_dict(matched, strict=False)
        print(f"[INFO] Discriminator: loaded {len(matched)}/{len(model_dict)} layers")

    # Freeze / unfreeze  logic                  
    def _set_backbone_frozen(self, frozen: bool):
        for name, param in self.net_g.named_parameters():
            is_head = any(k in name for k in self._HEAD_KEYWORDS)
            param.requires_grad = is_head if frozen else True

    def freeze_backbone(self):
        self._set_backbone_frozen(True)
        print("[INFO] Generator backbone frozen — only head layers will update")

    def unfreeze_backbone(self):
        self._set_backbone_frozen(False)
        print("[INFO] Generator backbone unfrozen — all layers will update")

    # Forward                                  
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # RRDBNet expects 3-channel input; TIR is single-channel
        x3 = x.repeat(1, 3, 1, 1)
        sr = self.net_g(x3)
        if sr.shape[1] != 1:
            sr = sr[:, :1, :, :]
        return sr

    # Shared metric computation (val / test — no GAN)                
    def _compute_metrics(self, sr: torch.Tensor, hr: torch.Tensor,
                         hr_mask: torch.Tensor, stage: str):
        """
        Shared PSNR / SSIM / MAE computation used by val and test steps.
        Mirrors swinir.py crop_to_valid_bbox strategy exactly.
        """
        sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
        hr_crop, _         = self.crop_to_valid_bbox(hr, hr_mask)
        valid_mask         = (mask_crop > 0.5)

        if valid_mask.sum() == 0:
            return (
                torch.tensor(0.0, device=sr.device),
                torch.tensor(0.0, device=sr.device),
                torch.tensor(0.0, device=sr.device),
            )

        mae_val = (sr_crop - hr_crop).abs()[valid_mask].mean()

        sr_for_metric = sr_crop.clone()
        hr_for_metric = hr_crop.clone()
        # sr_for_metric[~valid_mask] = hr_for_metric[~valid_mask]
        sr_for_metric[~valid_mask] = hr_for_metric[~valid_mask].to(sr_for_metric.dtype)

        psnr_val = getattr(self, f"{stage}_psnr")(sr_for_metric, hr_for_metric)

        h, w = sr_crop.shape[-2:]
        if h < 11 or w < 11:
            ssim_val = torch.tensor(0.0, device=sr.device)
        else:
            ssim_val = getattr(self, f"{stage}_ssim")(sr_for_metric, hr_for_metric)

        return psnr_val, ssim_val, mae_val

    # Training step — manual GAN optimisation                        
    def training_step(self, batch, batch_idx):
        lr_img, hr_img, hr_mask = batch
        opt_g, opt_d = self.optimizers()

        sr_img = self(lr_img)          # (B, 1, H*4, W*4)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        sr = torch.nan_to_num(sr, nan=0.0, posinf=1e6, neginf=-1e6)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6)

        #  Masked L1 reconstruction 
        abs_err    = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        # Gradient loss 
        if self.hparams.lambda_grad > 0:
            grad_loss = gradient_loss(sr, hr, hr_mask)
        else:
            grad_loss = torch.tensor(0.0, device=sr.device)

        # Perceptual loss (VGG on 3-ch) 
        if self.perceptual_loss is not None and self.hparams.lambda_perceptual > 0:
            sr_3ch = sr.repeat(1, 3, 1, 1)
            hr_3ch = hr.repeat(1, 3, 1, 1)
            percep_loss, _ = self.perceptual_loss(sr_3ch, hr_3ch)
            percep_loss = torch.nan_to_num(percep_loss, nan=0.0)
        else:
            percep_loss = torch.tensor(0.0, device=sr.device)

        # Generator step 
        self.toggle_optimizer(opt_g)

        pred_fake  = self.net_d(sr)
        gan_loss_g = self.gan_loss(pred_fake, target_is_real=True, is_disc=False)

        loss_g = (
            recon_loss
            + self.hparams.lambda_grad * grad_loss
            + percep_loss                           # weight already inside PerceptualLoss
            + gan_loss_g                            # weight already inside GANLoss
        )
        loss_g = torch.nan_to_num(loss_g, nan=0.0)

        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Discriminator step 
        self.toggle_optimizer(opt_d)

        pred_real   = self.net_d(hr)
        loss_d_real = self.gan_loss(pred_real, target_is_real=True,  is_disc=True)

        pred_fake_d = self.net_d(sr.detach())
        loss_d_fake = self.gan_loss(pred_fake_d, target_is_real=False, is_disc=True)

        loss_d = (loss_d_real + loss_d_fake) * 0.5
        loss_d = torch.nan_to_num(loss_d, nan=0.0)

        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Metrics 
        with torch.no_grad():
            psnr_val, ssim_val, mae_val = self._compute_metrics(
                sr, hr, hr_mask, "train"
            )

        # Logging 
        recon_loss  = torch.nan_to_num(recon_loss,  nan=0.0)
        grad_loss   = torch.nan_to_num(grad_loss,   nan=0.0)

        self.log("train_loss",             loss_g,      on_epoch=True, prog_bar=True)
        self.log("train_recon_loss",       recon_loss,  on_epoch=True)
        self.log("train_grad_loss",        grad_loss,   on_epoch=True)
        self.log("train_perceptual_loss",  percep_loss, on_epoch=True)
        self.log("train_gan_loss_g",       gan_loss_g,  on_epoch=True)
        self.log("train_loss_g",           loss_g,      on_epoch=True)
        self.log("train_loss_d",           loss_d,      on_epoch=True, prog_bar=True)
        self.log("train_psnr",             psnr_val,    on_epoch=True, prog_bar=True)
        self.log("train_ssim",             ssim_val,    on_epoch=True, prog_bar=True)
        self.log("train_mae",              mae_val,     on_epoch=True, prog_bar=True)

        return loss_g

    # Validation step                                              
    def validation_step(self, batch, batch_idx):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        sr = torch.nan_to_num(sr, nan=0.0, posinf=1e6, neginf=-1e6)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6)

        abs_err    = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        grad_loss = (
            gradient_loss(sr, hr, hr_mask)
            if self.hparams.lambda_grad > 0
            else torch.tensor(0.0, device=sr.device)
        )

        loss = recon_loss + self.hparams.lambda_grad * grad_loss
        loss = torch.nan_to_num(loss, nan=0.0)

        with torch.no_grad():
            psnr_val, ssim_val, mae_val = self._compute_metrics(
                sr, hr, hr_mask, "val"
            )

        self.log("val_loss",       loss,      on_epoch=True, prog_bar=True)
        self.log("val_recon_loss", recon_loss, on_epoch=True)
        self.log("val_grad_loss",  grad_loss,  on_epoch=True)
        self.log("val_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
        self.log("val_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
        self.log("val_mae",        mae_val,    on_epoch=True, prog_bar=True)

    # Test step                                                         
    def test_step(self, batch, batch_idx):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        sr = torch.nan_to_num(sr, nan=0.0, posinf=1e6, neginf=-1e6)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6)

        abs_err    = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        grad_loss = (
            gradient_loss(sr, hr, hr_mask)
            if self.hparams.lambda_grad > 0
            else torch.tensor(0.0, device=sr.device)
        )

        loss = recon_loss + self.hparams.lambda_grad * grad_loss
        loss = torch.nan_to_num(loss, nan=0.0)

        with torch.no_grad():
            psnr_val, ssim_val, mae_val = self._compute_metrics(
                sr, hr, hr_mask, "test"
            )

        self.log("test_loss",       loss,       on_epoch=True, prog_bar=True)
        self.log("test_recon_loss", recon_loss, on_epoch=True)
        self.log("test_grad_loss",  grad_loss,  on_epoch=True)
        self.log("test_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
        self.log("test_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
        self.log("test_mae",        mae_val,    on_epoch=True, prog_bar=True)

    # -------------- Optimiser -------------------------
    def configure_optimizers(self):
        # Generator: backbone vs head split 
        backbone_params, head_params = [], []
        for name, p in self.net_g.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in self._HEAD_KEYWORDS):
                head_params.append(p)
            else:
                backbone_params.append(p)

        opt_g = optim.Adam(
            [
                {"params": backbone_params,
                 "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
                {"params": head_params,
                 "lr": self.hparams.learning_rate},
            ],
            betas=(0.9, 0.99),
            weight_decay=1e-6,
        )

        opt_d = optim.Adam(
            self.net_d.parameters(),
            lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
            betas=(0.9, 0.99),
            weight_decay=1e-6,
        )

        # Scheduler monitors val_psnr — applied to generator only
        scheduler_g = optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode="max", 
            factor=0.5, 
            patience=self.hparams.patience
        )

        return (
            [opt_g, opt_d],
            [{"scheduler": scheduler_g, "monitor": "val_psnr"}],
        )