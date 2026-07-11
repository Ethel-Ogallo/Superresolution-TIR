# scripts/models/real_esrgan.py

"""
RealESRGAN baseline for TIR Super-Resolution.

"""

import os
import torch
import torch.optim as optim
import lightning.pytorch as pl

from basicsr.archs import rrdbnet_arch, discriminator_arch
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss

from scripts.utils.loss import masked_l1
from scripts.utils.metrics import compute_metrics


class RealESRGANModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        pretrained_d_path=None,
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        learning_rate=1e-4,
        d_lr_scale=1.0,
        n_feats=64,
        n_blocks=23,
        lambda_perceptual=1.0,
        lambda_adversarial=0.1,
    ):

        super().__init__()
        self.save_hyperparameters()
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE = data_range
        self.DATA_MIN = data_min


        # ---------------- Generator ----------------
        self.net_g = rrdbnet_arch.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=n_feats,
            num_block=n_blocks,
            num_grow_ch=32,
            scale=4,
        )

        if pretrained_path:
            self._load_generator(pretrained_path)

        # ---------------- Discriminator ----------------

        self.net_d = discriminator_arch.UNetDiscriminatorSN(
            num_in_ch=3,
            num_feat=64,
            skip_connection=True,
        )

        if pretrained_d_path:
            self._load_discriminator(pretrained_d_path)

        self.automatic_optimization = False

        # ---------------- Losses ----------------

        self.gan_loss = GANLoss(
            gan_type="vanilla",
            real_label_val=1.0,
            fake_label_val=0.0,
            loss_weight=lambda_adversarial,
        )

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

    # ------------------------------------------------
    # Forward
    # ------------------------------------------------

    def forward(self, lr):
        return self.net_g(lr)

    # ------------------------------------------------
    # Normalization
    # ------------------------------------------------
    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean

    # ------------------------------------------------
    # Training
    # ------------------------------------------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()

        lr = batch["lr"]
        hr = batch["hr"]
        hr_mask = batch["hr_mask"]

        sr = self(lr)

        # only thermal channel: denormalize to physical (°C) space before computing loss
        sr_tir, hr_tir = self.denormalize(sr[:,0:1]), self.denormalize(hr[:, 0:1])

        # ---------------- Generator ----------------
        self.toggle_optimizer(opt_g)

        recon_loss = masked_l1(sr_tir, hr_tir, hr_mask)

        perceptual_loss, _ = self.perceptual_loss(sr,hr)

        pred_fake = self.net_d(sr)

        gan_loss_g = self.gan_loss(
            pred_fake,
            target_is_real=True,
            is_disc=False
        )

        loss_g = ( recon_loss + perceptual_loss + gan_loss_g)

        self.manual_backward(loss_g)

        opt_g.step()
        opt_g.zero_grad()

        self.untoggle_optimizer(opt_g)

        # ---------------- Discriminator ----------------
        self.toggle_optimizer(opt_d)
        pred_real = self.net_d(hr)
        loss_d_real = self.gan_loss( pred_real, True, True)

        pred_fake = self.net_d(sr.detach())
        loss_d_fake = self.gan_loss( pred_fake, False, True)
        
        loss_d = 0.5 * ( loss_d_real + loss_d_fake)
        self.manual_backward(loss_d)

        opt_d.step()
        opt_d.zero_grad()

        self.untoggle_optimizer(opt_d)

        self.log_dict({
            "train/loss_g": loss_g,
            "train/recon": recon_loss,
            "train/perceptual": perceptual_loss,
            "train/gan_g": gan_loss_g,
            "train/gan_d": loss_d,
        },
        on_epoch=True,
        prog_bar=True)

        return loss_g

    # ------------------------------------------------
    # Validation
    # ------------------------------------------------
    @torch.no_grad()
    def validation_step(self,batch,batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self(lr)

        metrics = compute_metrics(
            self,
            self.denormalize(sr[:,0:1]),
            self.denormalize(hr[:,0:1]),
            batch["hr_mask"],
            "val",
            batch.get("water_mask")
        )

        self.log_dict(
            {
                f"val/{k}":v
                for k,v in metrics.items()
                if v is not None
            },
            on_epoch=True
        )

    # ------------------------------------------------
    # Test
    # ------------------------------------------------
    @torch.no_grad()
    def test_step(self,batch,batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self(lr)

        metrics = compute_metrics(
            self,
            self.denormalize(sr[:,0:1]),
            self.denormalize(hr[:,0:1]),
            batch["hr_mask"],
            "test",
            batch.get("water_mask")
        )

        self.log_dict(
            {
                f"test/{k}":v
                for k,v in metrics.items()
                if v is not None
            },
            on_epoch=True
        )


    # ------------------------------------------------
    # Optimizers
    # ------------------------------------------------
    def configure_optimizers(self):
        opt_g = optim.Adam(
            self.net_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9,0.99)
        )
        sch_g = optim.lr_scheduler.ReduceLROnPlateau(
            opt_g,
            mode="max",
            factor=0.5,
            patience=5
        )
        opt_d = optim.Adam(
            self.net_d.parameters(),
            lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
            betas=(0.9,0.99)
        )

        return (
            [opt_g,opt_d],
            [{"scheduler": sch_g, "monitor":"val/water_mae"}]
        )


    # ------------------------------------------------
    # Loading
    # ------------------------------------------------
    def _load_generator(self,path):
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        state_dict = ckpt.get( "params_ema", ckpt.get("params",ckpt))
        self.net_g.load_state_dict(state_dict, strict=False)
        print("[INFO] Generator loaded")

    def _load_discriminator(self,path):
        ckpt = torch.load(
            path,
            map_location="cpu",
            weights_only=False
        )
        state_dict = ckpt.get( "params", ckpt)
        self.net_d.load_state_dict( state_dict, strict=False)
        print("[INFO] Discriminator loaded")