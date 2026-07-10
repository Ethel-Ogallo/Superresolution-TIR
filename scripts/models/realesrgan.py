# scripts/models/real_esrgan.py

import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import rrdbnet_arch, discriminator_arch
from basicsr.losses.gan_loss import GANLoss
from basicsr.losses.basic_loss import PerceptualLoss
from torchmetrics.image import LearnedPerceptualImagePatchSimilarity

from scripts.utils.metrics import compute_metrics
from scripts.utils.loss import masked_l1


# AuxProjection
class AuxProjection(nn.Module):
    def __init__(self, aux_chans=23, out_chans=3):
        super().__init__()
        in_chans = 1 + aux_chans + 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 48, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(48, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 1),
        )

    def forward(self, tir, aux, time):
        return self.proj(torch.cat([tir, aux, time], dim=1))


# RealESRGAN module
class RealESRGANModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str      = None,
        pretrained_d_path: str    = None,
        hr_mean: float            = None,
        hr_std: float             = None,
        data_range: float         = None,
        data_min: float           = None,
        learning_rate: float      = 1e-4,
        d_lr_scale: float         = 1.0,
        n_feats: int              = 64,
        n_blocks: int             = 23,
        lambda_perceptual: float  = 1.0,
        lambda_adversarial: float = 0.1,
        adaptation_strategy: str  = "projection",
        aux_chans: int            = None,
        input_init: str           = "pretrained_mean",  # consistent with SwinIR naming
        freeze_backbone: bool     = False,               # default True for stability
        phase: int                = 2,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.DATA_RANGE     = data_range
        self.DATA_MIN       = data_min
        self.strategy       = adaptation_strategy
        self.aux_chans      = aux_chans
        self.hr_mean        = hr_mean
        self.hr_std         = hr_std
        self.freeze_backbone = freeze_backbone

        # only relevant for direct strategy
        self.input_init = input_init if adaptation_strategy == "direct" else None

        if aux_chans is None:
            raise ValueError("aux_chans must be provided")

        self.accum_steps = 4
        
        # ---------------- generator --------------------------
        # NOTE: always build with num_in_ch=3, matching the pretrained checkpoint's
        # native shape — this is the same pattern the SPADE script uses. Expanding
        # conv_first happens AFTER pretrained weights are loaded (see below), so the
        # mean-based init for the new channels is derived from real pretrained
        # features, not from a randomly-initialized layer.
        self.net_g = rrdbnet_arch.RRDBNet(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=n_feats,
            num_block=n_blocks,
            num_grow_ch=32,
            scale=4,
        )

        # load pretrained generator FIRST, while conv_first is still 3-channel
        if pretrained_path:
            self._load_pretrained_g(pretrained_path)

        # ---------------- strategy heads --------------------------
        self.proj = None

        if adaptation_strategy == "projection":
            self.proj = AuxProjection(aux_chans=aux_chans, out_chans=3)

        elif adaptation_strategy == "direct":
            # expand AFTER loading pretrained, so the new channels' init is derived
            # from the actual pretrained conv_first weights (fixes the previous
            # ordering bug where expansion happened before loading).
            self._expand_direct_input_layer()

        # output head: 3ch → 1ch
        self.out_head = nn.Conv2d(3, 1, 1)

        # ---------------- discriminator --------------------------
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

        # ----------- losses -----------------
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


        # ----- LPIPS Metric Initialization -----
        self.lpips_fn = LearnedPerceptualImagePatchSimilarity(net_type='vgg', normalize=True)
        self.lpips_fn.eval()
        for p in self.lpips_fn.parameters(): p.requires_grad = False

    # setup print
    def _print_setup(self):
        print("\n================ REALESRGAN SETUP ================")
        print(f"Strategy        : {self.strategy}")
        print(f"Input channels  : {self.in_chans}")
        print(f"Input init      : {self.input_init}")
        print(f"Freeze backbone : {self.freeze_backbone}")
        print("==================================================\n")

    # direct input layer expansion
    def _expand_direct_input_layer(self):
        old      = self.net_g.conv_first
        in_chans = 1 + self.aux_chans + 2  # TIR + aux + time channels

        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=(old.bias is not None),
        )

        with torch.no_grad():
            if self.input_init == "pretrained_mean":
                avg = old.weight.mean(dim=1, keepdim=True)
                new.weight[:] = avg.repeat(1, in_chans, 1, 1)

            elif self.input_init == "gaussian":
                nn.init.normal_(new.weight, mean=0.0, std=0.02)

            elif self.input_init == "xavier":
                nn.init.xavier_uniform_(new.weight)

            elif self.input_init == "he":
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")

            elif self.input_init == "partial_preserve":
                tir_init = old.weight.mean(dim=1, keepdim=True)
                new.weight[:, 0:1] = tir_init
                aux_init = old.weight.mean(dim=1, keepdim=True)
                noise = torch.randn_like(new.weight[:, 1:]) * 0.01
                new.weight[:, 1:] = aux_init + noise

            else:
                raise ValueError(f"Unknown input_init: {self.input_init}")

            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.net_g.conv_first = new
        print(f"[INFO] conv_first expanded 3 → {in_chans} ({self.input_init}, post-pretrained-load)")

    # -------------- freezing ----------------
    def _apply_freezing(self):
        if not self.freeze_backbone:
            return

        # freeze full generator backbone
        for p in self.net_g.parameters():
            p.requires_grad = False

        # for direct strategy: unfreeze conv_first so new channels can learn
        if self.strategy == "direct":
            for p in self.net_g.conv_first.parameters():
                p.requires_grad = True

    # ------------------ denormalise -----------
    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean

    # ---------------- forward ----------------
    def forward(self, batch):
        lr  = batch["lr"]
        aux = batch.get("aux", None)
        time  = batch["time_channels"]

        if self.strategy == "projection":
            x = self.proj(lr, aux, time)
        elif self.strategy == "direct":
            x = torch.cat([lr, aux, time], dim=1)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

        out = self.net_g(x)
        return self.out_head(out)

    # ---------------- training step ----------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        if batch_idx % self.accum_steps == 0:
            opt_g.zero_grad()
            opt_d.zero_grad()
            
        sr, hr = self(batch), batch["hr"]
        # sr_phys, hr_phys = sr, hr[:, 0:1]

        recon_loss = masked_l1(
                    sr[:, 0:1],
                    hr[:, 0:1],
                    batch["hr_mask"],
                )
 
        valid_mask = batch["hr_mask"][:, 0:1].float()
        sr_masked = sr * valid_mask
        hr_masked = hr[:, 0:1] * valid_mask

        # -------- Optimize Generator --------
        self.toggle_optimizer(opt_g)

        percep = self.perceptual_loss(sr_masked.repeat(1, 3, 1, 1), 
                                      hr_masked.repeat(1, 3, 1, 1))[0] if self.perceptual_loss else sr.sum() * 0.0
        gan_g = self.gan_loss(self.net_d(sr_masked.repeat(1, 3, 1, 1)), True, False) if self.hparams.lambda_adversarial > 0 else sr.sum() * 0.0

        loss_g = (recon_loss + percep + gan_g) / self.accum_steps
        self.manual_backward(loss_g)

        if (batch_idx + 1) % self.accum_steps == 0:
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_g.step()
        self.untoggle_optimizer(opt_g)

        # -------- Optimize Discriminator --------
        if self.hparams.lambda_adversarial > 0:
            self.toggle_optimizer(opt_d)
            loss_d = 0.5 * (
                self.gan_loss(self.net_d(hr_masked.repeat(1, 3, 1, 1)), True, True) +
                self.gan_loss(self.net_d(sr_masked.detach().repeat(1, 3, 1, 1)), False, True)
            )
            loss_d = loss_d / self.accum_steps
            self.manual_backward(loss_d)

            if (batch_idx + 1) % self.accum_steps == 0:
                self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
                opt_d.step()
            self.untoggle_optimizer(opt_d)
        else:
            loss_d = torch.tensor(0.0, device=sr.device)

        # ----- Logs --------
        self.log_dict({
            "train/loss_g":     loss_g * self.accum_steps,
            "train/recon":      recon_loss,
            "train/perceptual": percep,
            "train/gan_g":      gan_g,
            "train/gan_d":      loss_d * self.accum_steps,
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_g

    # -------------- Validation Steps --------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sr, hr = self(batch), batch["hr"]
        valid_mask = batch["hr_mask"][:, 0:1].float()
        sr_masked = sr * valid_mask
        hr_masked = hr[:, 0:1] * valid_mask

        recon_loss = masked_l1(
                        sr[:,0:1],
                        hr[:,0:1],
                        batch["hr_mask"],
                    )
        
        val_percep = self.perceptual_loss(sr_masked.repeat(1, 3, 1, 1), 
                                          hr_masked.repeat(1, 3, 1, 1))[0] if self.perceptual_loss else torch.tensor(0.0, device=sr.device)

        if self.hparams.lambda_adversarial > 0:
            val_gan_g = self.gan_loss(self.net_d(sr_masked.repeat(1, 3, 1, 1)), True, False)
            val_gan_d = 0.5 * (
                self.gan_loss(self.net_d(hr_masked.repeat(1, 3, 1, 1)), True, True) +
                self.gan_loss(self.net_d(sr_masked.detach().repeat(1, 3, 1, 1)), False, True)
            )
        else:
            val_gan_g = val_gan_d = torch.tensor(0.0, device=sr.device)

        self.log_dict({
            "val/loss_g":     recon_loss + val_percep + val_gan_g,
            "val/recon":      recon_loss,
            "val/perceptual": val_percep,
            "val/gan_g":      val_gan_g,
            "val/gan_d":      val_gan_d,
        }, on_step=False, on_epoch=True)
        
        # Computes metrics dict 
        metrics = compute_metrics(self, 
                                  self.denormalize(sr), 
                                  self.denormalize(hr[:, 0:1]), 
                                  batch["hr_mask"], 
                                  "val", 
                                  batch.get("water_mask"))
        
        self.log_dict(
            {f"val/{k}": v for k, v in metrics.items() if v is not None},
            on_step=False, on_epoch=True,
        )
        
        sch = self.lr_schedulers()
        if sch is not None:
            sch.step(self.trainer.callback_metrics.get("val/water_mae", 1.0))

    # -------------- Test Steps --------------
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        sr, hr = self(batch), batch["hr"]

        metrics = compute_metrics(self, 
                                  self.denormalize(sr), 
                                  self.denormalize(hr[:, 0:1]), 
                                  batch["hr_mask"], 
                                  "test", 
                                  batch.get("water_mask"))
        
        self.log_dict(
            {f"test/{k}": v for k, v in metrics.items() if v is not None},
            on_step=False, on_epoch=True,
        )

    # ---------------- optimiser ----------------
    def configure_optimizers(self):
        self._apply_freezing()

        # generator params
        g_params = [p for p in self.net_g.parameters() if p.requires_grad]
        g_params += list(self.out_head.parameters())

        if self.proj is not None:
            g_params += list(self.proj.parameters())

        opt_g = optim.Adam(
            g_params,
            lr=self.hparams.learning_rate,
            betas=(0.9, 0.99),
        )
        opt_d = optim.Adam(
            self.net_d.parameters(),
            lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
            betas=(0.9, 0.99),
        )
        sch_g = optim.lr_scheduler.ReduceLROnPlateau(
            opt_g, mode="min", factor=0.5, patience=5
        )
        return (
            [opt_g, opt_d],
            [{"scheduler": sch_g, "monitor": "val/water_mae"}],
        )

    # ---------------- pretrained loading ----------------
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