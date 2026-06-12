"""
real_esrgan.py — RealESRGAN Lightning Module for TIR Super-Resolution
This module handles TIR-SISR with a configurable SPADE bypass and comprehensive 
multi-task logging for physics-constrained evaluation.
"""

import torch
import torch.nn as nn
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
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        learning_rate: float = 1e-4,
        d_lr_scale: float = 1.0,
        n_feats: int = 64,
        n_blocks: int = 23,
        lambda_perceptual: float = 1.0,
        lambda_adversarial: float = 0.1,
        lambda_nw: float = 1.0,  
        lambda_w: float = 1.0,
        lambda_g: float = 0.1,  
        aux_chans: int = 23,
        use_spade: bool = True,  # Controls architecture path: True for SPADE, False for Baseline
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        # Physical normalization and domain constants
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE = data_range
        self.DATA_MIN = data_min
        self.lambda_nw = lambda_nw
        self.lambda_w = lambda_w
        self.lambda_g = lambda_g
        self.use_spade = use_spade
        self.aux_chans = aux_chans
        self.total_in_ch = 1 + aux_chans + 3 # 1 (TIR) + Aux (23) + Time (3)
        
        # KEY STABILITY: Virtual batch sizing multiplier
        self.accum_steps = 4

        # ----------- Architecture Initialization -----------
        rrdb = rrdbnet_arch.RRDBNet(num_in_ch=3, 
                                    num_out_ch=3, 
                                    num_feat=n_feats, 
                                    num_block=n_blocks, 
                                    num_grow_ch=32, 
                                    scale=4)
        
        if pretrained_path: self._load_weights(rrdb, pretrained_path, "generator")
        
        # Modify input layer to accept the expanded multi-channel auxiliary stack
        self._expand_conv_first(rrdb, self.total_in_ch)

        # SPADE wrapper - the bypass will route around this if use_spade=False
        self.net_g = RRDBNetWithSPADE(rrdb_net=rrdb, 
                                      n_feats=n_feats, 
                                      seg_nc_mid=aux_chans, 
                                      seg_nc_hr=aux_chans)
        self.out_head = nn.Conv2d(3, 1, 1)
        
        # Discriminator for adversarial training
        self.net_d = discriminator_arch.UNetDiscriminatorSN(num_in_ch=3, 
                                                            num_feat=64, 
                                                            skip_connection=True)
        
        if pretrained_d_path: self._load_weights(self.net_d, 
                                                 pretrained_d_path, 
                                                 "discriminator")

        # Required to take manual control over dual-optimizer GAN step updates
        self.automatic_optimization = False 

        # ----- Loss Configurations -----
        # ANALYSIS FIX: real_label_val changed to 0.9 (Label Smoothing).
        # This keeps the discriminator from hitting absolute certainty (1.0),
        # preserving ongoing gradient flow so it doesn't drop to 0 loss and stall training.
        self.gan_loss = GANLoss(gan_type="vanilla", 
                                real_label_val=0.9, 
                                fake_label_val=0.0, 
                                loss_weight=lambda_adversarial)
        
        # Pretrained VGG19 network handles structural composition penalties
        self.perceptual_loss = PerceptualLoss(
            layer_weights={"conv1_2": 0.1, "conv2_2": 0.1, "conv3_4": 1.0, "conv4_4": 1.0, "conv5_4": 1.0},
            vgg_type="vgg19", 
            use_input_norm=True, 
            range_norm=False, 
            perceptual_weight=lambda_perceptual,
            style_weight=0.0, 
            criterion="l1"
        ) if lambda_perceptual > 0 else None

        if self.perceptual_loss:
            for p in self.perceptual_loss.parameters(): p.requires_grad = False

    # -------------- Forward and Loss Computation --------------
    # adds the time channels to the input stack and routes through the appropriate architecture path based on use_spade flag
    def _build_time_channels(self, batch, H, W):
        B = batch["lr"].shape[0]
        device = batch["lr"].device
        def tile(v): return v[:, None, None, None].expand(B, 1, H, W)
        return torch.cat([tile(batch["lr_time"] / 24.0), tile(batch["time_gap_hours"] / 24.0), tile(batch["date_gap_days"] / 365.0)], dim=1).to(device)

    def forward(self, batch):
        lr = batch["lr"]
        x = torch.cat([lr, 
                    batch["aux_lr"], 
                    self._build_time_channels(batch, lr.shape[2], lr.shape[3])], dim=1)
        
        if self.use_spade:
            gen_out = self.net_g(x, batch["aux_mid"], batch["aux_hr"])
        else:
            gen_out = self.net_g(x, None, None)
            
        return self.out_head(gen_out)

    # -------------- Training Steps --------------
    def training_step(self, batch, batch_idx):
        opt_g, opt_d = self.optimizers()
        
        if batch_idx % self.accum_steps == 0:
            opt_g.zero_grad()
            opt_d.zero_grad()
            
        sr, hr = self(batch), batch["hr"]
        sr_phys, hr_phys = self.denormalize(sr), self.denormalize(hr[:, 0:1])

        # 1. Physics loss works on valid pixels 
        loss_dict = combined_loss(sr_phys, 
                                  hr_phys, 
                                  batch["hr_mask"], 
                                  batch["water_mask"], 
                                  self.lambda_nw, 
                                  self.lambda_w, 
                                  self.lambda_g, 
                                  batch["time_gap_hours"].mean().item(), 
                                  batch["date_gap_days"].mean().item()
                                )
        
        # 2. Extractvalid pixels mask [B, 1, H, W]
        valid_mask = batch["hr_mask"][:, 0:1].float()
        
        # 3. Apply mask to isolate the true data stream
        sr_masked = sr * valid_mask
        hr_masked = hr[:, 0:1] * valid_mask
        
        # 4. Structural Perceptual Loss (VGG) on valid pixels only
        percep = self.perceptual_loss(sr_masked.repeat(1, 3, 1, 1), hr_masked.repeat(1, 3, 1, 1))[0] if self.perceptual_loss else sr.sum() * 0.0
        
        # -------Optimize Generator------
        self.toggle_optimizer(opt_g)
        gan_g = self.gan_loss(self.net_d(sr_masked.repeat(1, 3, 1, 1)), True, False)
        
        loss_g = (loss_dict["loss_total"] + percep + gan_g) / self.accum_steps
        self.manual_backward(loss_g)
        
        if (batch_idx + 1) % self.accum_steps == 0: 
            self.clip_gradients(opt_g, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_g.step()
        self.untoggle_optimizer(opt_g)

        # --------Optimize Discriminator--------
        self.toggle_optimizer(opt_d)
        loss_d = 0.5 * (self.gan_loss(self.net_d(hr_masked.repeat(1, 3, 1, 1)), True, True) + 
                        self.gan_loss(self.net_d(sr_masked.detach().repeat(1, 3, 1, 1)), False, True))
        
        loss_d = loss_d / self.accum_steps
        self.manual_backward(loss_d)
        
        if (batch_idx + 1) % self.accum_steps == 0: 
            self.clip_gradients(opt_d, gradient_clip_val=0.5, gradient_clip_algorithm="norm")
            opt_d.step()
        self.untoggle_optimizer(opt_d)

        # ----- Logs --------
        self.log_dict({
            "train/loss_g": loss_g * self.accum_steps, 
            "train/recon": loss_dict["loss_total"], 
            "train/perceptual": percep, 
            "train/gan_g": gan_g, 
            "train/gan_d": loss_d * self.accum_steps, 
            "train/loss_nw": loss_dict["loss_nw"], 
            "train/loss_water": loss_dict["loss_water"], 
            "train/loss_grad": loss_dict["loss_grad"], 
            "train/w_loss_nw": loss_dict["loss_nw"] * self.lambda_nw,
            "train/w_loss_water": loss_dict["loss_water"] * self.lambda_w,
            "train/w_loss_grad": loss_dict["loss_grad"] * self.lambda_g
        }, on_step=False, on_epoch=True, prog_bar=True)
        
        return loss_g

    # -------------- Validation Steps --------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sr, hr = self(batch), batch["hr"]
        valid_mask = batch["hr_mask"][:, 0:1].float()
        
        sr_masked = sr * valid_mask
        hr_masked = hr[:, 0:1] * valid_mask

        loss_dict = combined_loss(self.denormalize(sr), 
                                  self.denormalize(hr[:, 0:1]), 
                                  batch["hr_mask"], 
                                  batch["water_mask"], 
                                  self.lambda_nw, 
                                  self.lambda_w, 
                                  self.lambda_g
                                  )
        
        val_percep = self.perceptual_loss(sr_masked.repeat(1, 3, 1, 1), hr_masked.repeat(1, 3, 1, 1))[0] if self.perceptual_loss else 0.0
        val_gan_g = self.gan_loss(self.net_d(sr_masked.repeat(1, 3, 1, 1)), True, False)
        val_gan_d = 0.5 * (self.gan_loss(self.net_d(hr_masked.repeat(1, 3, 1, 1)), True, True) + 
                           self.gan_loss(self.net_d(sr_masked.detach().repeat(1, 3, 1, 1)), False, True))

        self.log_dict({
            "val/loss_g": loss_dict["loss_total"] + val_percep + val_gan_g, 
            "val/recon": loss_dict["loss_total"], 
            "val/perceptual": val_percep,
            "val/gan_g": val_gan_g, 
            "val/gan_d": val_gan_d, 
            "val/loss_nw": loss_dict["loss_nw"],
            "val/loss_water": loss_dict["loss_water"], 
            "val/loss_grad": loss_dict["loss_grad"],
            "val/w_loss_nw": loss_dict["loss_nw"] * self.lambda_nw,
            "val/w_loss_water": loss_dict["loss_water"] * self.lambda_w,
            "val/w_loss_grad": loss_dict["loss_grad"] * self.lambda_g
        }, on_step=False, on_epoch=True)
        
        metrics = compute_metrics(self, 
                                  self.denormalize(sr), 
                                  self.denormalize(hr[:, 0:1]), 
                                  batch["hr_mask"], "val", 
                                  batch.get("water_mask"))
        
        self.log_dict({f"val/{k}": v for k, v in metrics.items() if v is not None}, on_step=False, on_epoch=True)
    
    # -------------- Optimizer Configuration --------------
    def configure_optimizers(self):
        g_params = list(self.net_g.parameters()) + list(self.out_head.parameters())
        
        # MOMENTUM FIX: Dropped beta1 down from 0.9 to 0.5.
        # This restrains historical step momentum, stopping the models from over-correcting
        # and generating sharp, erratic oscillation spikes.
        opt_g = optim.Adam(g_params, 
                           lr=self.hparams.learning_rate, 
                           betas=(0.5, 0.999))
        opt_d = optim.Adam(self.net_d.parameters(), 
                           lr=self.hparams.learning_rate * self.hparams.d_lr_scale,
                           betas=(0.5, 0.999))
        return [opt_g, opt_d], []

    # -------------- Utility functions ----------------     
    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean
    
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