"""
edsr.py — EDSR Lightning module for TIR super-resolution (x4).
- Backbone adapted from BasicSR's EDSR implementation.
- Uses FiLM-based auxiliary conditioning.
- Training/validation/test steps compute masked L1 + metrics.

Fixes applied:
  Bug 3 — __init__ now accepts aux_channels (total = cont + lulc) directly,
           so train.py's detected count is used instead of a stale hparam sum.
  FiLM   — forward passes target_size so gamma/beta always match res spatially.
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import edsr_arch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.metrics import shared_step
from scripts.utils.film_aux import AuxFiLM


class EDSRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        mean=0.0,
        std=1.0,
        aux_mean=None,
        aux_std=None,
        learning_rate=1e-4,
        bb_lr_scale=0.1,
        patience=5,
        n_feats=64,
        n_blocks=16,
        freeze_backbone=True,
        lambda_grad=0.1,
        data_range=70.0,
        use_aux=False,
        fusion_mode="film",  # "film" or "concat"
        # FIX 3 — accept the total detected channel count from train.py
        aux_channels=None,
        # kept as fallback when aux_channels is not supplied (e.g. manual init)
        aux_cont_channels=9,
        num_lulc_classes=11,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.use_aux = use_aux
        self.DATA_RANGE = float(data_range)
        self.aux_mean = aux_mean
        self.aux_std = aux_std

        # ---------------- backbone ----------------
        self.body = edsr_arch.EDSR(
            num_in_ch=3,
            num_out_ch=1,
            num_feat=n_feats,
            num_block=n_blocks,
            upscale=4,
            res_scale=0.1,
            img_range=1.0,
        )

        # ---------------- pretrained loading ----------------
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] Training from scratch")

        # Replace head
        self.body.conv_last = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1)
        self.body.mean = torch.zeros(1, 1, 1, 1)

        self._set_backbone_frozen(freeze_backbone)

        # ---------------- FiLM ----------------
        # if self.use_aux:
        #     # Prefer the explicitly-detected total; fall back to sum of parts
        #     self.aux_channels = (
        #         aux_channels
        #         if aux_channels is not None
        #         else aux_cont_channels + num_lulc_classes
        #     )
        #     print(f"[INFO] FiLM enabled | aux_channels = {self.aux_channels}")

        #     self.film = AuxFiLM(
        #         aux_channels=self.aux_channels,
        #         feat_channels=n_feats,
        #         # scale_factor=4,
        #     )
        # else:
        #     self.film = None
        #     self.aux_channels = 0
        #     print("[INFO] FiLM disabled")
        # edsr.py — __init__, replace FiLM block with a fusion conv

        # ---------------- FiLM or Concat fusion ----------------
        if self.use_aux:
            self.aux_channels = (
                aux_channels if aux_channels is not None
                else aux_cont_channels + num_lulc_classes
            )

            if fusion_mode == "film":
                print(f"[INFO] FiLM fusion enabled | aux_channels = {self.aux_channels}")
                self.film = AuxFiLM(
                    aux_channels=self.aux_channels,
                    feat_channels=n_feats,
                )
                self.aux_encoder = None
                self.fusion = None

            else:  # concat
                print(f"[INFO] Concat fusion enabled | aux_channels = {self.aux_channels}")
                self.film = None
                self.aux_encoder = nn.Sequential(
                    nn.Conv2d(self.aux_channels, n_feats, 3, padding=1, bias=False),
                    nn.InstanceNorm2d(n_feats, affine=True),
                    nn.ReLU(inplace=True),
                )
                self.fusion = nn.Conv2d(n_feats * 2, n_feats, 1, bias=False)

        else:
            self.film = None
            self.aux_encoder = None
            self.fusion = None
            self.aux_channels = 0
            print("[INFO] No AUX fusion")

        # ---------------- metrics ----------------
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ---------------- forward ----------------
    # def forward(self, lr, aux=None):
    #     # Repeat single channel to 3 for pretrained RGB backbone
    #     if lr.shape[1] == 1:
    #         lr = lr.repeat(1, 3, 1, 1)

    #     x   = self.body.conv_first(lr)
    #     res = self.body.body(x)
    #     res = self.body.conv_after_body(res) + x
    #     res = self.body.upsample(res)          # now at HR spatial size

    #     if self.use_aux and aux is not None:
    #         if aux.shape[1] != self.aux_channels:
    #             raise RuntimeError(
    #                 f"AUX channel mismatch: model expects {self.aux_channels}, "
    #                 f"got {aux.shape[1]}"
    #             )
    #         # FiLM fix — pass exact spatial size of res so shapes always match
    #         target_size = (res.shape[-2], res.shape[-1])
    #         gamma, beta = self.film(aux, target_size)
    #         res = gamma * res + beta

    #     return self.body.conv_last(res)
    # edsr.py — forward, replace FiLM modulation with concat+fusion

    def forward(self, lr, aux=None):
        if lr.shape[1] == 1:
            lr = lr.repeat(1, 3, 1, 1)

        x   = self.body.conv_first(lr)
        res = self.body.body(x)
        res = self.body.conv_after_body(res) + x
        res = self.body.upsample(res)

        if self.use_aux and aux is not None:
            if aux.shape[1] != self.aux_channels:
                raise RuntimeError(
                    f"AUX mismatch: expected {self.aux_channels}, got {aux.shape[1]}"
                )
            if self.film is not None:
                # FiLM mode
                target_size = (res.shape[-2], res.shape[-1])
                gamma, beta = self.film(aux, target_size)
                res = gamma * res + beta

            else:
                # Concat mode
                aux_feat = self.aux_encoder(aux)
                if aux_feat.shape[-2:] != res.shape[-2:]:
                    aux_feat = F.interpolate(
                        aux_feat, size=res.shape[-2:],
                        mode="bilinear", align_corners=False
                    )
                res = self.fusion(torch.cat([res, aux_feat], dim=1))

        return self.body.conv_last(res)

    # ---------------- helpers ----------------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    # ---------------- pretrained loading ----------------
    def _load_pretrained(self, path: str):
        print(f"[INFO] Loading pretrained: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)

        def remap_key(k):
            if k.startswith(("tail", "sub_mean", "add_mean")):
                return None
            if k.startswith("head.0."):
                return k.replace("head.0.", "conv_first.")
            if k.startswith("body."):
                if "body.16." in k:
                    return k.replace("body.16.", "conv_after_body.")
                k = k.replace(".body.0.", ".conv1.")
                k = k.replace(".body.2.", ".conv2.")
                return k
            return None

        model_dict = self.body.state_dict()
        matched = {}

        for k, v in state_dict.items():
            mk = remap_key(k)
            if mk is None or mk not in model_dict:
                continue
            if v.shape == model_dict[mk].shape:
                matched[mk] = v

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)} layers / {len(model_dict)} total")

    # ---------------- freezing logic ----------------
    def _set_backbone_frozen(self, frozen: bool):
        for name, p in self.body.named_parameters():
            p.requires_grad = (
                (not frozen)
                or ("conv_last" in name)
                or ("upsample" in name)
            )

    # ---------------- steps ----------------
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # ---------------- optimizer ----------------
    def configure_optimizers(self):
        backbone, head = [], []

        for name, p in self.body.named_parameters():
            if not p.requires_grad:
                continue
            if "conv_last" in name:
                head.append(p)
            else:
                backbone.append(p)

        param_groups = [
            {"params": backbone,
             "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
            {"params": head,
             "lr": self.hparams.learning_rate},
        ]

        # if self.use_aux and self.film is not None:
        #     param_groups.append({
        #         "params": self.film.parameters(),
        #         "lr": self.hparams.learning_rate,
        #     })
        if self.use_aux:
            if self.film is not None:
                param_groups.append({
                    "params": self.film.parameters(),
                    "lr": self.hparams.learning_rate,
                })
            else:
                param_groups.append({
                    "params": list(self.aux_encoder.parameters()) +
                            list(self.fusion.parameters()),
                    "lr": self.hparams.learning_rate,
                })

        optimizer = optim.Adam(param_groups, weight_decay=1e-6)

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=self.hparams.patience,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_psnr"},
        }