import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step
import torch.nn.functional as F


# --------------- AUX PROJECTION ------------------
class AuxProjection(nn.Module):
    def __init__(self, aux_chans=19, out_chans=3):
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


# ---------- FEATURE FUSION ---------
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


# ----------- SWINIR MODULE -----------
class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        adaptation_strategy="projection",
        aux_chans=None,
        lambda_grad=0.0,
        lambda_water=0.0,
        water_weight=0.0,
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        freeze_backbone=False,
        freeze_mode="none",   # none | body | body+first
        input_init="pretrained_mean",
        **kwargs
    ):
        super().__init__()

        self.save_hyperparameters()

        self.strategy = adaptation_strategy
        self.aux_chans = aux_chans
        self.freeze_backbone = freeze_backbone
        self.freeze_mode = freeze_mode

        # only keep input_init for direct strategy
        self.input_init = (
            input_init if adaptation_strategy == "direct"
            else None
        )

        self.lambda_grad = lambda_grad
        self.lambda_water = lambda_water
        self.water_weight = water_weight
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE   = data_range
        self.DATA_MIN     = data_min

        if aux_chans is None:
            raise ValueError("aux_chans must be provided")

        # --------- BACKBONE ---------
        body_in_chans = 1 + aux_chans if adaptation_strategy == "direct" else 3

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

        # -------------- STRATEGY HEADS ----------------
        self.proj = self.proj_out = None
        self.aux_encoder = self.fusion = self.fusion_out = None
        self.direct_out = None

        if adaptation_strategy == "projection":
            self.proj = AuxProjection(aux_chans=aux_chans)
            self.proj_out = nn.Conv2d(3, 1, 1)

        elif adaptation_strategy == "direct":
            self.direct_out = nn.Conv2d(1 + aux_chans, 1, 1)

        elif adaptation_strategy == "fusion":
            self.aux_encoder = nn.Sequential(
                nn.Conv2d(aux_chans, 64, 3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(64, embed_dim, 1),
            )
            self.fusion = FeatureFusion(embed_dim=embed_dim)
            self.fusion_out = nn.Conv2d(3, 1, 1)

        # DIRECT EXPANSION
        if adaptation_strategy == "direct":
            self._expand_direct_input_layer()

        # PRETRAIN LOAD
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        self._print_setup()

    # PRINT CONFIG
    def _print_setup(self):
        print("\n================ MODEL SETUP ================")
        print(f"Strategy       : {self.strategy}")
        print(f"Freeze backbone : {self.freeze_backbone}")
        print(f"Freeze mode     : {self.freeze_mode}")
        print("============================================\n")

    # ---------------- FREEZING LOGIC --------------------
    def _apply_freezing(self):

        if not self.freeze_backbone:
            return

        # freeze full backbone
        for p in self.body.parameters():
            p.requires_grad = False

        # selective unfreeze
        if self.freeze_mode == "body+first":
            for p in self.body.conv_first.parameters():
                p.requires_grad = True

        elif self.freeze_mode == "body":
            pass

        print("\n[INFO] FREEZING SUMMARY")
        for n, p in self.body.named_parameters():
            print(f"{n:40s} | {'TRAIN' if p.requires_grad else 'FROZEN'}")

    # ------------- DIRECT EXPANSION -----------------
    # def _expand_direct_input_layer(self):
    #     old = self.body.conv_first
    #     in_chans = 1 + self.aux_chans

    #     new = nn.Conv2d(
    #         in_chans,
    #         old.out_channels,
    #         old.kernel_size,
    #         old.stride,
    #         old.padding,
    #         bias=(old.bias is not None),
    #     )

    #     with torch.no_grad():
    #         new.weight.zero_()

    #         # ---- TIR channel ----
    #         # repeat RGB pretrained filters into single-channel form
    #         # (collapse RGB → 1 channel by averaging, then repeat is NOT needed here)
    #         tir_init = old.weight.mean(dim=1, keepdim=True)
    #         new.weight[:, 0:1] = tir_init

    #         # ---- AUX channels ----
    #         # repeat SAME initialization for all aux channels
    #         # (strong weight sharing assumption at initialization)
    #         aux_init = old.weight.mean(dim=1, keepdim=True)
    #         new.weight[:, 1:] = aux_init.repeat(1, in_chans - 1, 1, 1)

    #         if old.bias is not None:
    #             new.bias.copy_(old.bias)

    #     self.body.conv_first = new
    #     print(f"[INFO] conv_first expanded 3 - {in_chans} (REPEAT INIT)")

    def _expand_direct_input_layer(self):
        
        if self.strategy != "direct":
            return

        if self.input_init is None:
            raise ValueError("input_init must be set for direct strategy")

        old = self.body.conv_first
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
            if self.input_init == "pretrained_mean":

                avg = old.weight.mean(dim=1,keepdim=True)
                new.weight[:] = avg.repeat(1,in_chans,1,1)

            # GAUSSIAN
            elif self.input_init == "gaussian":

                nn.init.normal_(
                    new.weight,
                    mean=0.0,
                    std=0.02
                )

            # XAVIER
            elif self.input_init == "xavier":
                nn.init.xavier_uniform_(new.weight)

            # HE / KAIMING
            elif self.input_init == "he":
                nn.init.kaiming_normal_(new.weight, mode="fan_out", nonlinearity="relu")

            elif self.input_init == "partial_preserve":
                # TIR channel: strong pretrained prior
                tir_init = old.weight.mean(dim=1, keepdim=True)
                new.weight[:, 0:1] = tir_init
                # AUX channels: weakly structured init (NOT identical copies)
                aux_init = old.weight.mean(dim=1, keepdim=True)
                noise = torch.randn_like(new.weight[:, 1:]) * 0.01
                new.weight[:, 1:] = aux_init + noise

            else:
                raise ValueError(f"Unknown input_init: {self.input_init}")

            if old.bias is not None:
                new.bias.copy_(old.bias)


        self.body.conv_first = new

        print(
            f"[INFO] conv_first expanded "
            f"3 - {in_chans} "
            f"({self.input_init})")

    # -------------- PRETRAIN LOADING ----------------
    def _load_pretrained(self, path):

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt)

        matched = {
            k: v for k, v in state.items()
            if k in self.body.state_dict()
            and v.shape == self.body.state_dict()[k].shape
        }

        self.body.load_state_dict(matched, strict=False)

        print(f"[INFO] Loaded {len(matched)} / {len(self.body.state_dict())} weights")

    
    # ---------- DENORMALIZE -------------
    def denormalize(self, x, mean=None, std=None):
        if mean is None or std is None:
            return x
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std = torch.tensor(std, device=x.device, dtype=x.dtype)
        return x * std + mean

    # --------------- FORWARD --------------------
    def forward(self, batch):

        lr = batch["lr"]
        aux = batch.get("aux", None)

        if self.strategy == "direct":
            x = torch.cat([lr, aux], dim=1)
            out = self.body(x)
            return self.direct_out(out)

        if self.strategy == "projection":
            x = self.proj(lr, aux)
            out = self.body(x)
            return self.proj_out(out)

        if self.strategy == "fusion":
            x = lr.repeat(1, 3, 1, 1)

            feat = self.body.conv_first(x)
            aux_feat = self.aux_encoder(aux)

            aux_feat = F.interpolate(aux_feat, feat.shape[-2:])

            feat = self.fusion(feat, aux_feat)

            feat = self.body.forward_features(feat)
            feat = self.body.conv_after_body(feat)
            feat = self.body.conv_before_upsample(feat)
            feat = self.body.upsample(feat)
            feat = self.body.conv_last(feat)

            return self.fusion_out(feat)

    # --------------- TRAIN LOOP --------------------
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # --------------- OPTIMIZER & SCHEDULER --------------------
    def configure_optimizers(self):

        self._apply_freezing()

        params = []

        # backbone (if trainable)
        if not self.freeze_backbone:
            params.append({
                "params": self.body.parameters(),
                "lr": self.hparams.learning_rate * 0.1,
            })
        else:
            params.append({
                "params": [p for p in self.body.parameters() if p.requires_grad],
                "lr": self.hparams.learning_rate * 0.1,
            })

        # heads
        for m in [self.proj, self.proj_out, self.direct_out,
                  self.aux_encoder, self.fusion, self.fusion_out]:
            if m is not None:
                params.append({
                    "params": m.parameters(),
                    "lr": self.hparams.learning_rate,
                })

        opt = optim.Adam(params)

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_full_psnr"}
        }