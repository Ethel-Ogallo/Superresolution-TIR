import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


# -----------------------------
# Projection module (INPUT fusion)
# -----------------------------
class AuxProjection(nn.Module):
    def __init__(self, in_chans, out_chans=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 1)
        )
        self.tir_skip = nn.Conv2d(1, out_chans, kernel_size=1)

    def forward(self, x):
        tir = x[:, 0:1, :, :]
        fused = self.net(x)
        skip = self.tir_skip(tir)
        return fused + skip


# -----------------------------
# Fusion module (FEATURE fusion)
# -----------------------------
class AuxEncoder(nn.Module):
    def __init__(self, in_chans, out_chans):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, out_chans, 1),
            nn.GroupNorm(36, out_chans)      
        )

    def forward(self, x):
        return self.net(x)


class FeatureFusion(nn.Module):
    def __init__(self, embed_dim):
        super().__init__()
        self.fuse = nn.Sequential(
            nn.Conv2d(embed_dim * 2, embed_dim, 1),  # 360 → 180
            nn.ReLU(inplace=True),
            nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
        )

    def forward(self, feat, aux_feat):
        x = torch.cat([feat, aux_feat], dim=1)       # (B, 360, H, W)
        fused = self.fuse(x)                         # (B, 180, H, W)
        return fused + feat                          # TIR skip — feat always contributes

# -----------------------------
# SwinIR Module
# -----------------------------
class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        depths=None,
        num_heads=None,
        window_size=8,
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        data_range=None,
        hr_mean=None,
        hr_std=None,
        adaptation_strategy="direct",
        in_aux_chans=None,
        lambda_grad=0.0,
        lambda_water=0.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.adaptation_strategy = adaptation_strategy
        self.in_aux_chans = in_aux_chans

        self.DATA_RANGE = data_range
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.lambda_grad = lambda_grad
        self.lambda_water = lambda_water

        depths = depths or [6]*6
        num_heads = num_heads or [6]*6

        # -------------------------
        # backbone input channels
        # -------------------------
        if adaptation_strategy == "direct":
            assert in_aux_chans is not None, "DIRECT requires aux channels"
            in_chans = 1 + in_aux_chans
        else:
            in_chans = 3

        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=in_chans,
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

        # self.output_head = nn.Conv2d(3, 1, 1)
        self.output_head = nn.Conv2d(in_chans, 1, 1)

        # -------------------------
        # projection
        # -------------------------
        self.proj = None
        if adaptation_strategy == "projection" and in_aux_chans is not None:
            self.proj = AuxProjection(in_chans=1 + in_aux_chans, out_chans=3)

        # -------------------------
        # fusion
        # -------------------------
        self.aux_encoder = None
        self.fusion = None

        if adaptation_strategy == "fusion" and in_aux_chans is not None:
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(in_aux_chans, embed_dim, 3, padding=1),
                nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 1)
            )

            self.fusion = nn.Sequential(
                nn.Conv2d(embed_dim * 2, embed_dim, 1),
                nn.ReLU(),
                nn.Conv2d(embed_dim, embed_dim, 3, padding=1)
            )

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        if adaptation_strategy == "direct":
            self._adapt_input_layer(in_chans)

    # =====================================================
    # SAFE INPUT ADAPTATION (FIXED)
    # =====================================================
    def _adapt_input_layer(self, in_chans):
        old = self.body.conv_first

        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            kernel_size=old.kernel_size,
            stride=old.stride,
            padding=old.padding,
        ).to(old.weight.device)

        with torch.no_grad():

            # copy RGB channels safely
            min_ch = min(3, in_chans)
            new.weight[:, :min_ch] = old.weight[:, :min_ch]

            # initialize extras
            if in_chans > 3:
                mean = old.weight[:, :3].mean(dim=1, keepdim=True)
                new.weight[:, 3:] = mean.repeat(1, in_chans - 3, 1, 1)

            new.bias.copy_(old.bias)

        self.body.conv_first = new

    # =====================================================
    # FORWARD (CORRECTED)
    # =====================================================
    def forward(self, batch):

        lr = batch["lr"]
        aux = batch.get("aux", None)

        # -------------------------
        # BASELINE
        # -------------------------
        if aux is None or self.adaptation_strategy == "baseline":
            return self.output_head(self.body(lr))

        # -------------------------
        # INPUT BUILD
        # -------------------------
        x = torch.cat([lr, aux], dim=1)

        # -------------------------
        # DIRECT / PROJECTION
        # -------------------------
        if self.adaptation_strategy in ["direct", "projection"]:

            if self.adaptation_strategy == "projection":
                x = self.proj(x)
            sr = self.body(x)
            if sr.shape[1] != 3:
                sr = self.output_head(sr)
            return sr
        
        # -------------------------
        # FEATURE FUSION
        # -------------------------
        elif self.adaptation_strategy == "fusion":

            x_rgb = lr.repeat(1, 3, 1, 1)
            x_rgb = (x_rgb - self.body.mean.to(x_rgb.device)) * self.body.img_range

            x = self.body.conv_first(x_rgb)
            feat = self.body.forward_features(x)

            aux_feat = self.aux_encoder(aux)
            feat = self.fusion(torch.cat([feat, aux_feat], dim=1))

            x = self.body.conv_after_body(feat) + x
            x = self.body.conv_before_upsample(x)
            x = self.body.conv_last(self.body.upsample(x))

            x = x / self.body.img_range + self.body.mean.to(x.device)

            return self.output_head(x)

        else:
            raise ValueError(self.adaptation_strategy)

    # -------- Denormalization --------
    def denormalize(self, x):
        if self.hr_mean is None or self.hr_std is None:
            return x
        mean = torch.tensor(self.hr_mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(self.hr_std,  device=x.device, dtype=x.dtype)
        return x * std + mean

    # -----------------------------
    # TRAIN / VAL / TEST
    # -----------------------------
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # -----------------------------
    # OPTIMIZER
    # -----------------------------
    def configure_optimizers(self):
        param_groups = [
            {
                "params": self.body.parameters(),
                "lr": self.hparams.learning_rate * 0.1  # pretrained — conservative
            },
            {
                "params": self.output_head.parameters(),
                "lr": self.hparams.learning_rate
            },
        ]

        if self.proj is not None:
            param_groups.append({
                "params": self.proj.parameters(),
                "lr": self.hparams.learning_rate
            })

        if self.aux_encoder is not None:
            param_groups.append({
                "params": self.aux_encoder.parameters(),
                "lr": self.hparams.learning_rate
            })

        if self.fusion is not None:
            param_groups.append({
                "params": self.fusion.parameters(),
                "lr": self.hparams.learning_rate
            })

        opt = optim.Adam(param_groups)

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_full_psnr"
            }
        }
    # -----------------------------
    # PRETRAIN LOADING
    # -----------------------------
    def _load_pretrained(self, path):
        if not os.path.exists(path):
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt)
        model_dict = self.body.state_dict()

        matched = {
            k: v for k, v in state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        self.body.load_state_dict(matched, strict=False)