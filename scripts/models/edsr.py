# scripts/models/edsr.py
"""
EDSR Lightning module for TIR Super-Resolution.
Phase 1: zero-shot inference
Phase 2: fine-tuning with masked L1
Architecture untouched — 3ch in, 3ch out, metrics on ch0 only.
"""

import os
import torch
import torch.optim as optim
import lightning.pytorch as pl
import torch.nn.functional as F
from basicsr.archs import edsr_arch
from scripts.utils.metrics import compute_metrics
from scripts.utils.loss import masked_l1


class EDSRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str  = None,
        hr_mean: float        = None,
        hr_std: float         = None,
        learning_rate: float  = 1e-4,
        n_feats: int          = 64,
        n_blocks: int         = 16,
        data_range: float     = None,
        data_min: float       = None,
        # phase: int            = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE = data_range
        self.DATA_MIN = data_min


        self.body = edsr_arch.EDSR(
            num_in_ch=3,
            num_out_ch=3,
            num_feat=n_feats,
            num_block=n_blocks,
            upscale=4,
            res_scale=0.1,
            img_range=1.0,
        )

        if pretrained_path:
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] No pretrained path — random init")
    
    # forward pass
    def forward(self, lr):
        return self.body(lr)           # (B, 3, 256, 256)

    def denormalize(self, t):
        return t * self.hr_std + self.hr_mean

    # ------------- training ------------
    def training_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]
        hr_mask = batch["hr_mask"]

        sr = self(lr)

        # TIR only
        sr = sr[:, 0:1]
        hr = hr[:, 0:1]

        # denormalize to physical (°C) space before computing loss
        sr_phys = self.denormalize(sr)
        hr_phys = self.denormalize(hr)

        loss = masked_l1(sr_phys, hr_phys, hr_mask)

        self.log(
            "train/loss",
            loss,
            prog_bar=True,
            on_step=False,
            on_epoch=True
        )

        return loss

    # -------------- validation ------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self(lr)

        # only TIR channel
        sr = sr[:,0:1]
        hr = hr[:,0:1]

        # convert back to degrees C
        sr = self.denormalize(sr)
        hr = self.denormalize(hr)

        metrics = compute_metrics(
            module=self,
            sr=sr,
            hr=hr,
            hr_mask=batch["hr_mask"],
            stage="val",
            water_mask=batch.get("water_mask")
        )

        for name, value in metrics.items():

            if value is not None:
                self.log(
                    f"val/{name}",
                    value,
                    on_step=False,
                    on_epoch=True
                )

    # -------------- test ------------
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        lr = batch["lr"]
        hr = batch["hr"]

        sr = self(lr)

        sr = sr[:,0:1]
        hr = hr[:,0:1]

        sr = self.denormalize(sr)
        hr = self.denormalize(hr)

        metrics = compute_metrics(
            module=self,
            sr=sr,
            hr=hr,
            hr_mask=batch["hr_mask"],
            stage="test",
            water_mask=batch.get("water_mask")
        )

        for name, value in metrics.items():

            if value is not None:
                self.log(
                    f"test/{name}",
                    value,
                    on_step=False,
                    on_epoch=True
                )


    # ------------- optimizer and scheduler------------
    def configure_optimizers(self):
        opt = optim.Adam(
            self.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=1e-6
        )
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="min", factor=0.5, patience=5
        )
        return {"optimizer": opt,
                "lr_scheduler": {"scheduler": sch, "monitor": "val/water_mae"}}

    # Pretrained loading
    def _load_pretrained(self, path):
        if not os.path.exists(path):
            print(f"[WARNING] Pretrained not found: {path}")
            return
        print(f"[INFO] Loading EDSR pretrained: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)

        # EDSR checkpoint keys differ from BasicSR implementation
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
        matched    = {}
        for k, v in state_dict.items():
            mk = remap_key(k)
            if mk is None or mk not in model_dict:
                continue
            if v.shape == model_dict[mk].shape:
                matched[mk] = v

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} layers")