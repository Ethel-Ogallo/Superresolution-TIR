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
        water_weight=2.0,
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
        self.water_weight = water_weight
        self.strategy     = adaptation_strategy
        self.aux_chans    = aux_chans

        if aux_chans is None:
            raise ValueError("aux_chans must be provided")

        # =====================================================
        # BACKBONE
        # direct: SwinIR sees 21 channels in and 21 out
        # projection / fusion: SwinIR always sees 3 channels
        # =====================================================
        if adaptation_strategy == "direct":
            body_in_chans = 1 + aux_chans   # 21
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
        # STRATEGY MODULES + OUTPUT HEADS
        # Every strategy produces [B, 1, H*4, W*4] so that
        # shared_step's sr_img[:, 0:1] is always correct
        # =====================================================
        self.proj        = None
        self.proj_out    = None
        self.aux_encoder = None
        self.fusion      = None
        self.fusion_out  = None
        self.direct_out  = None

        if adaptation_strategy == "projection":
            # 4-layer CNN compresses (TIR + aux) → 3ch pseudo-RGB
            # SwinIR processes 3ch, then proj_out collapses 3 → 1 TIR channel
            self.proj     = AuxProjection(aux_chans=aux_chans)
            self.proj_out = nn.Conv2d(3, 1, kernel_size=1)

        elif adaptation_strategy == "direct":
            # SwinIR processes all 21 channels end to end
            # conv_first and conv_last train from scratch (shapes differ)
            # all transformer layers (layers.0-5) still load from pretrained
            # direct_out collapses 21 → 1 TIR channel
            self.direct_out = nn.Conv2d(1 + aux_chans, 1, kernel_size=1)

        elif adaptation_strategy == "fusion":

            # AUX encoder → map AUX into SwinIR feature space
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(aux_chans, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, embed_dim, 1),
            )

            self.fusion = FeatureFusion(embed_dim=embed_dim)

            # collapse final SR output (3 → 1)
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

        # shape-matched loading
        # for direct: conv_first and conv_last are skipped automatically
        # because their shapes differ (21ch vs pretrained 3ch)
        # for projection / fusion: all 550 weights load
        matched = {
            k: v for k, v in state.items()
            if k in self.body.state_dict()
            and v.shape == self.body.state_dict()[k].shape
        }

        self.body.load_state_dict(matched, strict=False)

        skipped = sorted(set(self.body.state_dict().keys()) - set(matched.keys()))
        print(f"[INFO] Loaded  : {len(matched)} weights")
        if skipped:
            print(f"[INFO] Skipped : {skipped}")

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
    # All strategies return [B, 1, H*4, W*4]
    # shared_step's sr_img[:, 0:1] is correct for all strategies
    # =========================================================
    def forward(self, batch):

        lr  = batch["lr"]
        aux = batch.get("aux", None)

        # -------------------------
        # DIRECT
        # SwinIR sees all 21 channels end to end
        # direct_out collapses 21 → 1
        # -------------------------
        if self.strategy == "direct":
            x   = torch.cat([lr, aux], dim=1)   # [B, 21, H,    W   ]
            out = self.body(x)                   # [B, 21, H*4,  W*4 ]
            out = self.direct_out(out)           # [B,  1, H*4,  W*4 ]
            return out

        # -------------------------
        # PROJECTION
        # AuxProjection compresses (TIR + aux) → 3ch
        # SwinIR processes 3ch
        # proj_out collapses 3 → 1
        # -------------------------
        if self.strategy == "projection":
            x   = self.proj(lr, aux)             # [B,  3, H,    W   ]
            out = self.body(x)                   # [B,  3, H*4,  W*4 ]
            out = self.proj_out(out)             # [B,  1, H*4,  W*4 ]
            return out

        # -------------------------
        # FUSION
        # SwinIR processes TIR only (3ch repeated)
        # aux injected at feature level via fusion module
        # fusion_out collapses 3 → 1
        # -------------------------
        if self.strategy == "fusion":

            # Prepare SwinIR input (TIR only)
            x = lr.repeat(1, 3, 1, 1)

            # Shallow feature extraction (IMPORTANT FIX)
            feat = self.body.conv_first(x)

            #  AUX feature encoding
            aux_feat = self.aux_encoder(aux)

            # match spatial resolution if needed
            if aux_feat.shape[-2:] != feat.shape[-2:]:
                aux_feat = F.interpolate(
                    aux_feat,
                    size=feat.shape[-2:],
                    mode="bilinear",
                    align_corners=False
                )


            feat = self.fusion(feat, aux_feat)  #  EARLY FEATURE FUSION (FAIR VERSION)
            feat = self.body.forward_features(feat)
            feat = self.body.conv_after_body(feat)
            feat = self.body.conv_before_upsample(feat)
            feat = self.body.upsample(feat)
            feat = self.body.conv_last(feat)

            out = self.fusion_out(feat)

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
    # OPTIMIZER
    # =========================================================
    def configure_optimizers(self):

        # backbone at 10x lower lr — pretrained weights fine-tune slowly
        param_groups = [
            {"params": self.body.parameters(),
             "lr": self.hparams.learning_rate * 0.1},
        ]

        # all strategy-specific modules train at full lr
        for module in [
            self.proj,
            self.proj_out,
            self.direct_out,
            self.aux_encoder,
            self.fusion,
            self.fusion_out,
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
            }
        }