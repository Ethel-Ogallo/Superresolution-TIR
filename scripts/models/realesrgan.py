# scripts/models/real_esrgan.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import rrdbnet_arch, discriminator_arch
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss
from scripts.utils.metrics import shared_step
from scripts.utils.loss import masked_l1


# AuxProjection
class AuxProjection(nn.Module):
    def __init__(self, aux_chans=20, out_chans=3):
        super().__init__()
        in_chans = 1 + aux_chans
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 1),
        )

    def forward(self, tir, aux):
        return self.proj(torch.cat([tir, aux], dim=1))


#  RealESRGAN module
class RealESRGANModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str      = None,
        pretrained_d_path: str    = None,
        hr_mean: float            = 0.0,
        hr_std: float             = 1.0,
        learning_rate: float      = 1e-4,
        d_lr_scale: float         = 1.0,
        n_feats: int              = 64,
        n_blocks: int             = 23,
        data_range: float         = None,
        data_min: float           = None,
        lambda_perceptual: float  = 1.0,
        lambda_adversarial: float = 0.1,
        adaptation_strategy: str  = "projection",
        aux_chans: int            = None,
        direct_init_mode: str     = "mean",
    ):
        super().__init__()
        self.save_hyperparameters()

        self.DATA_RANGE = float(data_range)
        self.DATA_MIN   = float(data_min) if data_min is not None else 0.0
        self.strategy   = adaptation_strategy
        self.aux_chans  = aux_chans

        if aux_chans is None:
            raise ValueError("aux_chans must be provided")

        # generator 
        body_in_chans = (1 + aux_chans) if adaptation_strategy == "direct" else 3

        self.net_g = rrdbnet_arch.RRDBNet(
            num_in_ch=body_in_chans,
            num_out_ch=3,
            num_feat=n_feats,
            num_block=n_blocks,
            num_grow_ch=32,
            scale=4,
        )

        # strategy heads 
        self.proj = None

        if adaptation_strategy == "projection":
            self.proj = AuxProjection(aux_chans=aux_chans, out_chans=3)
        elif adaptation_strategy == "direct":
            self._expand_direct_input_layer(init_mode=direct_init_mode)

        # load pretrained generator 
        if pretrained_path:
            self._load_pretrained_g(pretrained_path)

        # output head: 3ch → 1ch
        self.out_head = nn.Conv2d(3, 1, 1)

        # discriminator
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
            for p in self.perceptual_loss.parameters():
                p.requires_grad = False
        else:
            self.perceptual_loss = None

        self._print_setup()

    def _print_setup(self):
        print("\n================ REALESRGAN SETUP ================")
        print(f"Strategy        : {self.strategy}")
        print(f"Aux channels    : {self.aux_chans}")
        print(f"Direct init mode: {self.hparams.direct_init_mode}")
        print("==================================================\n")

    # input layer expansion 
    def _expand_direct_input_layer(self, init_mode="mean"):
        old      = self.net_g.conv_first
        in_chans = 1 + self.aux_chans

        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=(old.bias is not None),
        )

        with torch.no_grad():
            # Different initialization strategies for new input channels
            # PRETRAINED MEAN + REPEAT
            if self.init_mode == "pretrained_mean":

                avg = old.weight.mean(dim=1,keepdim=True)
                new.weight[:] = avg.repeat(1,in_chans,1,1)

            # GAUSSIAN
            elif self.init_mode == "gaussian":

                nn.init.normal_(
                    new.weight,
                    mean=0.0,
                    std=0.02
                )

            # XAVIER
            elif self.init_mode == "xavier":
                nn.init.xavier_uniform_(new.weight)

            # HE / KAIMING
            elif self.init_mode == "he":
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")

            else:
                raise ValueError(f"Unknown init_mode: {self.init_mode}")

            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.net_g.conv_first = new

        print(
            f"[INFO] conv_first expanded "
            f"3 - {in_chans} "
            f"({self.init_mode})")

    # ── forward ──────────────────────────────────────────────────────────────
    def forward(self, batch):
        lr  = batch["lr"]
        aux = batch.get("aux", None)

        if self.strategy == "projection":
            x = self.proj(lr, aux)
        elif self.strategy == "direct":
            x = torch.cat([lr, aux], dim=1)

        out = self.net_g(x)
        return self.out_head(out)

    # ── denormalise ──────────────────────────────────────────────────────────
    def denormalize(self, t, mean=None, std=None):
        if mean is None or std is None:
            return t
        mean = torch.tensor(mean, device=t.device, dtype=t.dtype)
        std  = torch.tensor(std,  device=t.device, dtype=t.dtype)
        return t * std + mean

    # ── training step ────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        hr_img  = batch["hr"]
        hr_mask = batch["hr_mask"]

        opt_g, opt_d = self.optimizers()

        sr_img = self(batch)
        sr     = self.denormalize(sr_img, self.hparams.hr_mean, self.hparams.hr_std)
        hr     = self.denormalize(hr_img[:, 0:1], self.hparams.hr_mean, self.hparams.hr_std)
        sr     = torch.nan_to_num(sr, nan=0.0)
        hr     = torch.nan_to_num(hr, nan=0.0)

        # repeat to 3ch for discriminator + perceptual loss
        sr_3ch = sr_img.repeat(1, 3, 1, 1)
        hr_3ch = hr_img[:, 0:1].repeat(1, 3, 1, 1)

        # ── generator step ───────────────────────────────────────────────────
        self.toggle_optimizer(opt_g)

        recon_loss = masked_l1(sr, hr, hr_mask)

        if self.perceptual_loss is not None:
            percep_loss, _ = self.perceptual_loss(sr_3ch, hr_3ch)
            percep_loss    = torch.nan_to_num(percep_loss, nan=0.0)
        else:
            percep_loss = torch.tensor(0.0, device=sr.device)

        pred_fake  = self.net_d(sr_3ch)
        gan_loss_g = self.gan_loss(pred_fake, target_is_real=True, is_disc=False)

        loss_g = recon_loss + percep_loss + gan_loss_g
        loss_g = torch.nan_to_num(loss_g, nan=0.0)

        self.manual_backward(loss_g)
        opt_g.step()
        opt_g.zero_grad()
        self.untoggle_optimizer(opt_g)

        # ── discriminator step ───────────────────────────────────────────────
        self.toggle_optimizer(opt_d)

        pred_real   = self.net_d(hr_3ch)
        loss_d_real = self.gan_loss(pred_real,           target_is_real=True,  is_disc=True)
        pred_fake_d = self.net_d(sr_3ch.detach())
        loss_d_fake = self.gan_loss(pred_fake_d,         target_is_real=False, is_disc=True)
        loss_d      = (loss_d_real + loss_d_fake) * 0.5
        loss_d      = torch.nan_to_num(loss_d, nan=0.0)

        self.manual_backward(loss_d)
        opt_d.step()
        opt_d.zero_grad()
        self.untoggle_optimizer(opt_d)

        # ── logging ──────────────────────────────────────────────────────────
        self.log("train_loss",        loss_g,      on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train_loss_d",      loss_d,      on_step=True,  on_epoch=True, prog_bar=True)
        self.log("train_recon_loss",  recon_loss,  on_step=False, on_epoch=True)
        self.log("train_percep_loss", percep_loss, on_step=False, on_epoch=True)
        self.log("train_gan_loss_g",  gan_loss_g,  on_step=False, on_epoch=True)

        with torch.no_grad():
            from scripts.utils.metrics import compute_metrics
            compute_metrics(self, sr, hr, hr_mask, "train")

        return loss_g

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # ── optimiser ────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        g_params = list(self.net_g.parameters()) + list(self.out_head.parameters())
        if self.proj is not None:
            g_params += list(self.proj.parameters())

        opt_g = optim.Adam(
            g_params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
            weight_decay=1e-6,
        )
        opt_d = optim.Adam(
            self.net_d.parameters(),
            lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
            betas=(0.9, 0.99),
            weight_decay=1e-6,
        )
        sch_g = optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode="max", factor=0.5, patience=5
        )
        return (
            [opt_g, opt_d],
            [{"scheduler": sch_g, "monitor": "val_full_psnr"}],
        )

    # ── pretrained loading ───────────────────────────────────────────────────
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