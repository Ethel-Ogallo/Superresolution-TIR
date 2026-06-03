"""
real_esrgan.py — RealESRGAN Lightning Module for TIR Super-Resolution
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

from scripts.models.spade import RRDBNetWithSPADE
from scripts.utils.metrics import compute_metrics
from scripts.utils.loss import combined_loss


class RealESRGANModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_path: str = None,
        pretrained_d_path: str = None,
        hr_mean = None,
        hr_std = None,
        data_range = None,
        data_min = None,
        learning_rate: float = 1e-4,
        d_lr_scale: float = 1.0,
        n_feats: int = 64,
        n_blocks: int = 23,
        lambda_perceptual: float = 1.0,
        lambda_adversarial: float = 0.1,
        lambda_nw: float = 1.0,  
        lambda_w: float = 1.0,
        lambda_g: float = 0.1,  
        aux_chans: int = 20,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ----------------------------
        # constants
        # ----------------------------
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE = data_range
        self.DATA_MIN = data_min
        self.lambda_nw = lambda_nw
        self.lambda_w = lambda_w
        self.lambda_g = lambda_g
        self.aux_chans = aux_chans

        self.time_chans = 3
        self.total_in_ch = 1 + aux_chans + self.time_chans
        self.accum_steps = 4

        # ============================================================
        # GENERATOR & DISCRIMINATOR INITIALIZATION
        # ============================================================
        rrdb = rrdbnet_arch.RRDBNet(
            num_in_ch=3, num_out_ch=3, num_feat=n_feats, num_block=n_blocks, num_grow_ch=32, scale=4
        )
        if pretrained_path:
            self._load_weights(rrdb, pretrained_path, "generator")

        self._expand_conv_first(rrdb, self.total_in_ch)

        self.net_g = RRDBNetWithSPADE(
            rrdb_net=rrdb, n_feats=n_feats, seg_nc_mid=aux_chans, seg_nc_hr=aux_chans
        )
        self.out_head = nn.Conv2d(3, 1, 1)

        self.net_d = discriminator_arch.UNetDiscriminatorSN(
            num_in_ch=3, num_feat=64, skip_connection=True
        )
        if pretrained_d_path:
            self._load_weights(self.net_d, pretrained_d_path, "discriminator")

        self.automatic_optimization = False

        # ============================================================
        # LOSSES
        # ============================================================
        self.gan_loss = GANLoss(
            gan_type="vanilla", real_label_val=1.0, fake_label_val=0.0, loss_weight=lambda_adversarial
        )

        self.perceptual_loss = None
        if lambda_perceptual > 0:
            self.perceptual_loss = PerceptualLoss(
                layer_weights={"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0},
                vgg_type="vgg19", use_input_norm=True, range_norm=False, perceptual_weight=lambda_perceptual,
                style_weight=0.0, criterion="l1"
            )
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False

    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean

    def _build_time_channels(self, batch, H, W):
        B = batch["lr"].shape[0]
        device = batch["lr"].device
        def tile(v): return v[:, None, None, None].expand(B, 1, H, W)
        lr_time = batch["lr_time"] / 24.0
        gap_h = batch["time_gap_hours"] / 24.0
        gap_d = batch["date_gap_days"] / 365.0
        return torch.cat([tile(lr_time), tile(gap_h), tile(gap_d)], dim=1).to(device)

    def forward(self, batch):
        lr = batch["lr"]
        x = torch.cat([lr, batch["aux_lr"], self._build_time_channels(batch, lr.shape[2], lr.shape[3])], dim=1)
        return self.out_head(self.net_g(x, batch["aux_mid"], batch["aux_hr"]))

    # ============================================================
    # TRAINING STEP
    # ============================================================
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        # ----------------------------
        # Optimize Generator
        # ----------------------------
        self.toggle_optimizer(opt_g)
        sr = self(batch)
        hr = batch["hr"]

        sr_phys = self.denormalize(sr)
        hr_phys = self.denormalize(hr[:, 0:1])

        loss_dict = combined_loss(
            sr=sr_phys,
            hr=hr_phys,
            hr_mask=batch["hr_mask"],
            water_mask=batch["water_mask"],
            lambda_nw=self.lambda_nw,
            lambda_w=self.lambda_w,
            lambda_g=self.lambda_g,
            time_gap_hours=batch["time_gap_hours"].mean().item(),
            date_gap_days=batch["date_gap_days"].mean().item(),
        )
        recon_loss = loss_dict["loss_total"]

        if self.perceptual_loss is not None:
            percep = self.perceptual_loss(sr.repeat(1, 3, 1, 1), hr[:, 0:1].repeat(1, 3, 1, 1))[0]
        else:
            percep = sr.sum() * 0.0

        pred_fake = self.net_d(sr.repeat(1, 3, 1, 1))
        gan_g = self.gan_loss(pred_fake, True, False)

        loss_g = recon_loss + percep + gan_g
        self.manual_backward(loss_g)

        if (batch_idx + 1) % self.accum_steps == 0 or self.trainer.is_last_batch:
            opt_g.step()
            opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # ----------------------------
        # Optimize Discriminator
        # ----------------------------
        self.toggle_optimizer(opt_d)
        real = self.net_d(hr[:, 0:1].repeat(1, 3, 1, 1))
        fake = self.net_d(sr.detach().repeat(1, 3, 1, 1))
        loss_d = 0.5 * (self.gan_loss(real, True, True) + self.gan_loss(fake, False, True))
        self.manual_backward(loss_d)

        if (batch_idx + 1) % self.accum_steps == 0 or self.trainer.is_last_batch:
            opt_d.step()
            opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Updated physical tracking metrics for the new equation
        self.log("train/recon", recon_loss)
        self.log("train/perceptual", percep)
        self.log("train/gan_g", gan_g)
        self.log("train/gan_d", loss_d)
        self.log("train/loss_nw", loss_dict["loss_nw"])
        self.log("train/loss_water", loss_dict["loss_water"])
        self.log("train/loss_grad", loss_dict["loss_grad"])
        self.log("train/w_time", loss_dict["w_time"])

        return loss_g

    # ============================================================
    # VALIDATION STEP
    # ============================================================
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sr = self(batch)
        hr = batch["hr"]

        sr_phys = self.denormalize(sr)
        hr_phys = self.denormalize(hr[:, 0:1])

        loss_dict = combined_loss(
            sr=sr_phys,
            hr=hr_phys,
            hr_mask=batch["hr_mask"],
            water_mask=batch["water_mask"],
            lambda_nw=self.lambda_nw,
            lambda_w=self.lambda_w,
            lambda_g=self.lambda_g,
            time_gap_hours=batch["time_gap_hours"].mean().item(),
            date_gap_days=batch["date_gap_days"].mean().item(),
        )

        self.log("val/recon", loss_dict["loss_total"])
        self.log("val/loss_nw", loss_dict["loss_nw"])
        self.log("val/loss_water", loss_dict["loss_water"])
        self.log("val/loss_grad", loss_dict["loss_grad"])

        metrics = compute_metrics(
            module=self,
            sr=sr_phys,
            hr=hr_phys,
            hr_mask=batch["hr_mask"],
            stage="val",
            water_mask=batch.get("water_mask"),
            w_time=loss_dict["w_time"]
        )

        # LAND TRACKING (NON-WATER METRICS)
        self.log("val/land_psnr", metrics["land_psnr"])
        self.log("val/land_ssim", metrics["land_ssim"])
        self.log("val/land_mae", metrics["land_mae"])
        self.log("val/land_rmse", metrics["land_rmse"])

        # RIVER CHANNEL TRACKING (WATER METRICS)
        if metrics["water_mae"] is not None:
            self.log("val/water_psnr", metrics["water_psnr"], prog_bar=True)
            self.log("val/water_ssim", metrics["water_ssim"])
            self.log("val/water_mae", metrics["water_mae"], prog_bar=True) # Direct visibility
            self.log("val/water_rmse", metrics["water_rmse"])

        if "w_time" in metrics:
            self.log("val/w_time", metrics["w_time"])

        return None

    def configure_optimizers(self):
        g_params = list(self.net_g.parameters()) + list(self.out_head.parameters())
        opt_g = optim.Adam(g_params, lr=self.hparams.learning_rate, betas=(0.9, 0.99))
        opt_d = optim.Adam(self.net_d.parameters(), lr=self.hparams.learning_rate * self.hparams.d_lr_scale)
        return [opt_g, opt_d], []

    def _expand_conv_first(self, rrdb, in_ch):
        old = rrdb.conv_first
        new = nn.Conv2d(in_ch, old.out_channels, old.kernel_size, old.stride, old.padding)
        with torch.no_grad():
            mean = old.weight.mean(dim=1, keepdim=True)
            new.weight.copy_(mean.repeat(1, in_ch, 1, 1))
            if old.bias is not None: new.bias.copy_(old.bias)
        rrdb.conv_first = new

    def _load_weights(self, model, path, name):
        ckpt = torch.load(path, map_location="cpu")
        sd = ckpt.get("params_ema", ckpt.get("params", ckpt))
        msd = model.state_dict()
        filtered = {k: v for k, v in sd.items() if k in msd and v.shape == msd[k].shape}
        model.load_state_dict(filtered, strict=False)
        print(f"[INFO] {name} loaded {len(filtered)} params")