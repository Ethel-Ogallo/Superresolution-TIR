import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step
import torch.nn.functional as F


# =========================================================
# AUX PROJECTION (projection strategy only)
# =========================================================
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


# =========================================================
# FEATURE FUSION (fusion strategy only)
# =========================================================
class FeatureFusion(nn.Module):
    def __init__(self, embed_dim=180):
        super().__init__()

        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1),
        )

        self.gate = nn.Parameter(torch.tensor(0.7))

    def forward(self, tir_feat, aux_feat):
        return tir_feat + self.gate * self.fuse(
            torch.cat([tir_feat, aux_feat], dim=1)
        )


# =========================================================
# FiLM (TIME CONDITIONING MODULE)
# =========================================================
class FiLM(nn.Module):
    def __init__(self, cond_dim, embed_dim):
        super().__init__()

        self.net = nn.Sequential(
            nn.Linear(cond_dim, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 128),
            nn.ReLU(inplace=True),
            nn.Linear(128, 2 * embed_dim)
        )

        # keeps modulation stable at start of training
        self.scale = nn.Parameter(torch.tensor(0.3))

    def forward(self, cond):
        """
        cond: [B, cond_dim]
        returns:
            gamma, beta: [B, C]
        """

        out = self.net(cond)
        gamma, beta = torch.chunk(out, 2, dim=1)

        # stabilize early training
        gamma = torch.tanh(gamma) * self.scale + 1.0
        beta  = torch.tanh(beta) * self.scale

        return gamma, beta


# =========================================================
# SWINIR MODULE
# =========================================================
class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        data_range=None,
        data_min=None,
        hr_mean=None,
        hr_std=None,
        adaptation_strategy="projection",
        aux_chans=None,
        lambda_grad=0.0,
        lambda_water=0.0,
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.DATA_RANGE   = data_range
        self.DATA_MIN     = data_min
        self.hr_mean      = hr_mean
        self.hr_std       = hr_std
        self.lambda_grad  = lambda_grad
        self.lambda_water = lambda_water
        self.strategy     = adaptation_strategy
        self.aux_chans    = aux_chans

        if aux_chans is None:
            raise ValueError("aux_chans must be provided")

        # =====================================================
        # BACKBONE
        # =====================================================
        if adaptation_strategy == "direct":
            body_in_chans = 1 + aux_chans
        else:
            body_in_chans = 3

        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=body_in_chans,
            img_size=img_size,
            window_size=8,
            img_range=1.0,
            depths=[6] * 6,
            embed_dim=embed_dim,
            num_heads=[6] * 6,
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        # =====================================================
        # TIME CONDITIONING
        # =====================================================

        self.use_time = True
        self.time_mode = kwargs.get("time_mode", "none")

        if self.time_mode == "none":
            self.cond_dim = 0
        elif self.time_mode in ["date", "time"]:
            self.cond_dim = 1
        else:
            self.cond_dim = 2

        self.film = None

        if self.cond_dim > 0:
            self.film = FiLM(
                cond_dim=self.cond_dim,
                embed_dim=embed_dim
            )

        # =====================================================
        # STRATEGY MODULES
        # =====================================================
        self.proj        = None
        self.proj_out    = None
        self.aux_encoder = None
        self.fusion      = None
        self.fusion_out  = None
        self.direct_out  = None

        if adaptation_strategy == "projection":
            self.proj     = AuxProjection(aux_chans=aux_chans)
            self.proj_out = nn.Conv2d(3, 1, kernel_size=1)

        elif adaptation_strategy == "direct":
            self.direct_out = nn.Conv2d(1 + aux_chans, 1, kernel_size=1)

        elif adaptation_strategy == "fusion":
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(aux_chans, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, embed_dim, 1),
            )
            self.fusion     = FeatureFusion(embed_dim=embed_dim)
            self.fusion_out = nn.Conv2d(3, 1, kernel_size=1)

        # =====================================================
        # PRETRAIN
        # =====================================================
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    # =========================================================
    # PRETRAIN LOADER
    # =========================================================
    def _load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"[INFO] Pretrained path not found: {path}")
            return

        ckpt  = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt)

        matched = {
            k: v for k, v in state.items()
            if k in self.body.state_dict()
            and v.shape == self.body.state_dict()[k].shape
        }

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)} pretrained weights")

    # =========================================================
    # DENORMALIZE
    # =========================================================
    def denormalize(self, x):
        if self.hr_mean is None or self.hr_std is None:
            return x

        mean = torch.tensor(self.hr_mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(self.hr_std,  device=x.device, dtype=x.dtype)

        return x * std + mean

    # =========================================================
    # FORWARD
    # =========================================================
    def forward(self, batch):

        lr  = batch["lr"]
        aux = batch.get("aux", None)
        cond = batch.get("cond", None)   # [B, 6] time conditioning

        # =====================================================
        # DIRECT  (no FiLM — cond not used here)
        # =====================================================
        if self.strategy == "direct":
            x   = torch.cat([lr, aux], dim=1)
            out = self.body(x)
            out = self.direct_out(out)
            return out

        # =====================================================
        # PROJECTION + MULTI-SCALE FiLM
        # =====================================================
        if self.strategy == "projection":

            x = self.proj(lr, aux)

            feat = self.body.conv_first(x)

            # FiLM injection 
            if self.use_time and cond is not None and self.cond_dim > 0:

                gamma, beta = self.film(cond)

                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta  = beta.unsqueeze(-1).unsqueeze(-1)

                feat = gamma * feat + beta

            # SwinIR backbone
            feat = self.body.forward_features(feat)
            feat = self.body.conv_after_body(feat)
            feat = self.body.conv_before_upsample(feat)
            feat = self.body.upsample(feat)
            feat = self.body.conv_last(feat)

            out = self.proj_out(feat)

            return out

        # =====================================================
        # FUSION  
        # =====================================================
        if self.strategy == "projection":

            x = self.proj(lr, aux)

            feat = self.body.conv_first(x)

            if (
                self.use_time
                and self.film is not None
                and cond is not None
            ):

                gamma, beta = self.film(cond)

                gamma = gamma.unsqueeze(-1).unsqueeze(-1)
                beta = beta.unsqueeze(-1).unsqueeze(-1)

                feat = gamma * feat + beta

            feat = self.body.forward_features(feat)
            feat = self.body.conv_after_body(feat)
            feat = self.body.conv_before_upsample(feat)
            feat = self.body.upsample(feat)
            feat = self.body.conv_last(feat)

            out = self.proj_out(feat)

            return out

    # =========================================================
    # TRAINING / VALIDATION / TEST
    # =========================================================
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # =========================================================
    # OPTIMIZER AND SCHEDULER
    # =========================================================
    def configure_optimizers(self):

        param_groups = [
            # Backbone — lower LR (pretrained weights)
            {
                "params": self.body.parameters(),
                "lr": self.hparams.learning_rate * 0.1,
            },
        ]

        # New modules — full LR
        for module in [
            self.proj,
            self.proj_out,
            self.direct_out,
            self.aux_encoder,
            self.fusion,
            self.fusion_out,
            self.film,   
        ]:
            if module is not None:
                param_groups.append({
                    "params": module.parameters(),
                    "lr": self.hparams.learning_rate,
                })

        opt = optim.Adam(param_groups)

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_full_psnr",
            },
        }