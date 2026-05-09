# scripts/models/real_esrgan.py
"""
RealESRGAN Lightning module for TIR Super-Resolution.

Phase 1: zero-shot inference — pretrained generator only, no discriminator
Phase 2: full GAN fine-tuning — pretrained generator + pretrained discriminator
         masked L1 + perceptual + adversarial losses

Architecture untouched — 3ch in, 3ch out, metrics on ch0 only.
"""

import os
import torch
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import rrdbnet_arch, discriminator_arch
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss
from scripts.utils.metrics import shared_step
from scripts.utils.loss import masked_l1


class RealESRGANModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str     = None,
        pretrained_d_path: str   = None,
        hr_mean: float           = 0.0,
        hr_std: float            = 1.0,
        learning_rate: float     = 1e-4,
        d_lr_scale: float        = 1.0,
        n_feats: int             = 64,
        n_blocks: int            = 23,
        data_range: float        = None,
        phase: int               = 1,
        lambda_perceptual: float = 1.0,
        lambda_adversarial: float = 0.1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_RANGE = float(data_range)

        #  Generator + reconstruction loss (both phases)
        self.net_g = rrdbnet_arch.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=n_feats,
            num_block=n_blocks,
            num_grow_ch=32,
            scale=4,
        )

        if pretrained_path:
            self._load_pretrained_g(pretrained_path)
        else:
            print("[INFO] Generator — no pretrained path, random init")

        # Discriminator + GAN losses (Phase 2 only) 
        if phase == 2:
            self.automatic_optimization = False

            self.net_d = discriminator_arch.UNetDiscriminatorSN(
                num_in_ch=3,
                num_feat=64,
                skip_connection=True,
            )

            if pretrained_d_path:
                self._load_pretrained_d(pretrained_d_path)
            else:
                print("[INFO] Discriminator — no pretrained path, random init")

            self.gan_loss = GANLoss(
                gan_type="vanilla",
                real_label_val=1.0,
                fake_label_val=0.0,
                loss_weight=lambda_adversarial,
            )

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
                # VGG frozen — not trained
                for p in self.perceptual_loss.parameters():
                    p.requires_grad = False
            else:
                self.perceptual_loss = None

            print("[INFO] Phase 2 — GAN training enabled")
        else:
            self.net_d           = None
            self.gan_loss        = None
            self.perceptual_loss = None
            print("[INFO] Phase 1 — inference only, no discriminator")

    # Forward 
    def forward(self, lr):
        return self.net_g(lr)              # (B, 3, 256, 256)

    # Denormalise 
    def denormalize(self, t):
        mean = torch.tensor(self.hparams.hr_mean, device=t.device)
        std  = torch.tensor(self.hparams.hr_std,  device=t.device)
        return t * std + mean

    # Phase 1 steps — simple shared_step 
    def _phase1_step(self, batch, stage):
        return shared_step(self, batch, stage)

    # Phase 2 training — full GAN 
    def _phase2_training_step(self, batch, batch_idx):
        lr_img  = batch["lr"]
        hr_img  = batch["hr"]
        hr_mask = batch["hr_mask"]

        opt_g, opt_d = self.optimizers()
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img[:, 0:1])
        hr = self.denormalize(hr_img[:, 0:1])
        sr = torch.nan_to_num(sr, nan=0.0)
        hr = torch.nan_to_num(hr, nan=0.0)

        # Generator step 
        self.toggle_optimizer(opt_g)

        recon_loss = masked_l1(sr, hr, hr_mask)

        if self.perceptual_loss is not None:
            percep_loss, _ = self.perceptual_loss(sr_img, hr_img)
            percep_loss    = torch.nan_to_num(percep_loss, nan=0.0)
        else:
            percep_loss = torch.tensor(0.0, device=sr.device)

        pred_fake  = self.net_d(sr_img)
        gan_loss_g = self.gan_loss(pred_fake, target_is_real=True, is_disc=False)

        loss_g = recon_loss + percep_loss + gan_loss_g
        loss_g = torch.nan_to_num(loss_g, nan=0.0)

        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # Discriminator step 
        self.toggle_optimizer(opt_d)

        pred_real   = self.net_d(hr_img)
        loss_d_real = self.gan_loss(pred_real,   target_is_real=True,  is_disc=True)
        pred_fake_d = self.net_d(sr_img.detach())
        loss_d_fake = self.gan_loss(pred_fake_d, target_is_real=False, is_disc=True)
        loss_d      = (loss_d_real + loss_d_fake) * 0.5
        loss_d      = torch.nan_to_num(loss_d, nan=0.0)

        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # Logging 
        self.log("train_loss",        loss_g,      on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_loss_d",      loss_d,      on_step=True, on_epoch=True, prog_bar=True)
        self.log("train_recon_loss",  recon_loss,  on_step=False, on_epoch=True)
        self.log("train_percep_loss", percep_loss, on_step=False, on_epoch=True)
        self.log("train_gan_loss_g",  gan_loss_g,  on_step=False, on_epoch=True)

        # Metrics 
        with torch.no_grad():
            from scripts.utils.metrics import compute_metrics
            compute_metrics(self, sr, hr, hr_mask, "train")

        return loss_g

    # Step routing
    def training_step(self, batch, batch_idx):
        if self.hparams.phase == 2:
            return self._phase2_training_step(batch, batch_idx)
        return self._phase1_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # Optimiser
    def configure_optimizers(self):
        if self.hparams.phase == 1:
            # Phase 1 — single optimiser, not used for training
            opt = optim.Adam(
                self.net_g.parameters(),
                lr=self.hparams.learning_rate,
                weight_decay=1e-6
            )
            sch = optim.lr_scheduler.ReduceLROnPlateau(
                opt, mode="max", factor=0.5, patience=5
            )
            return {"optimizer": opt,
                    "lr_scheduler": {"scheduler": sch, "monitor": "val_psnr"}}

        # Phase 2 — two optimisers for GAN
        opt_g = optim.Adam(
            self.net_g.parameters(),
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-6
        )
        opt_d = optim.Adam(
            self.net_d.parameters(),
            lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
            betas=(0.9, 0.99),
            weight_decay=1e-6
        )
        sch_g = optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode="max", factor=0.5, patience=5
        )
        return (
            [opt_g, opt_d],
            [{"scheduler": sch_g, "monitor": "val_psnr"}],
        )

    # Pretrained loading
    def _load_pretrained_g(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Generator pretrained not found: {path}")
            return
        print(f"[INFO] Loading RealESRGAN generator: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))
        model_dict = self.net_g.state_dict()
        matched    = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        self.net_g.load_state_dict(matched, strict=False)
        print(f"[INFO] Generator: loaded {len(matched)}/{len(model_dict)} layers")

    def _load_pretrained_d(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Discriminator pretrained not found: {path}")
            return
        print(f"[INFO] Loading RealESRGAN discriminator: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params", ckpt)
        model_dict = self.net_d.state_dict()
        matched    = {
            k: v for k, v in state_dict.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }
        self.net_d.load_state_dict(matched, strict=False)
        print(f"[INFO] Discriminator: loaded {len(matched)}/{len(model_dict)} layers")