"""
edsr.py — EDSR Lightning module for TIR super-resolution (×4).

Architecture changes vs. the RGB baseline:
  - Input  : 1-channel TIR - repeated to 3 channels
  - Output : 3-channel temperature map
  - Pretrained weights loaded with key remapping
"""

import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import edsr_arch
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure


class EDSRModule(pl.LightningModule):
    """
    EDSR backbone adapted for single-channel TIR super-resolution.
    Hparams:
        pretrained_path   : path to EDSR_baseline_x4.pth (or None)
        mean / std        : dataset statistics for denormalisation
        learning_rate     : LR for unfrozen body params
        backbone_lr_scale : multiplier applied to backbone LR
        patience          : ReduceLROnPlateau + EarlyStopping patience
        n_feats           : number of feature channels
        n_blocks          : number of residual blocks
        freeze_backbone   : freeze body except conv_last + upsample layers
    """

    DATA_RANGE = 60.0  # °C — used by PSNR / SSIM

    def __init__(
        self,
        pretrained_path: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        learning_rate: float = 1e-4,
        backbone_lr_scale: float = 0.1,
        patience: int = 5,
        n_feats: int = 64,
        n_blocks: int = 16,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        self.save_hyperparameters()

        # ---------- Model architecture ----------
        # Input: 3-channel TIR repeated from 1-channel dataset
        # Output: 3-channel super-resolved TIR map
        self.body = edsr_arch.EDSR(
            num_in_ch=3,  
            num_out_ch=3, 
            num_feat=n_feats,
            num_block=n_blocks,
            upscale=4,
            res_scale=0.1,
            img_range=1.0,
            rgb_mean=(0.0, 0.0, 0.0),
        )

        # ---------- Load pretrained ----------
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            self.print("[INFO] No pretrained weights — training EDSR from scratch")

        # ---------- If output is single-channel ----------
        # self.body.conv_last = nn.Conv2d(n_feats, 1, kernel_size=3, padding=1) 
        # self.body.mean = torch.zeros(1, 1, 1, 1)  

        # ---------- Freezing logic ----------
        self._set_backbone_frozen(freeze_backbone)

        # ---------- Metrics ----------
        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # ---------- Utils ----------
    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        # Broadcast mean/std to 3 channels for repeated TIR input
        mean = torch.tensor(self.hparams.mean, device=t.device).view(1, 3, 1, 1)
        std = torch.tensor(self.hparams.std, device=t.device).view(1, 3, 1, 1)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        # Crop tensor to the tight bounding box of valid (mask=1) pixels
        m = mask.squeeze(1).any(dim=0)
        rows = m.any(dim=1).nonzero(as_tuple=True)[0]
        cols = m.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

    # ---------- Pretrained loader ----------
    def _load_pretrained(self, path: str):
        self.print(f"[INFO] Loading pretrained weights: {path}")
        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state_dict = ckpt.get("params", ckpt)

        # Map checkpoint keys to current model keys
        def remap_key(k: str):
            if k.startswith(("tail", "sub_mean", "add_mean")):
                return None
            if k.startswith("head.0."):
                return k.replace("head.0.", "conv_first.", 1)
            if k.startswith("body."):
                if k.startswith("body.16."):
                    return k.replace("body.16.", "conv_after_body.", 1)
                k = k.replace(".body.0.", ".conv1.", 1)
                k = k.replace(".body.2.", ".conv2.", 1)
                return k
            return None

        model_dict = self.body.state_dict()
        matched = {}
        for ckpt_key, v in state_dict.items():
            model_key = remap_key(ckpt_key)
            if model_key is None or model_key not in model_dict:
                continue
            if v.shape != model_dict[model_key].shape:
                continue
            matched[model_key] = v

        self.body.load_state_dict(matched, strict=False)
        self.print(f"[INFO] Matched {len(matched)} / {len(model_dict)} layers")

    # ---------- Freezing ----------
    def _set_backbone_frozen(self, frozen: bool):
        # If freeze_backbone=True, freeze all except conv_last and upsample
        for name, param in self.body.named_parameters():
            param.requires_grad = (
                (not frozen)
                or ("conv_last" in name)
                or ("upsample" in name)
            )

    # ---------- Forward ----------
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x)  

    # ---------- Shared step ----------
    def _shared_step(self, batch, stage: str):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)
        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        # Masked L1 loss: only valid HR pixels contribute
        abs_err = torch.abs(sr - hr) * hr_mask
        loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        with torch.no_grad():
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _ = self.crop_to_valid_bbox(hr, hr_mask)
            psnr_val = getattr(self, f"{stage}_psnr")(sr_crop * mask_crop, hr_crop * mask_crop)
            ssim_val = getattr(self, f"{stage}_ssim")(sr_crop, hr_crop)

        self.log(f"{stage}_loss", loss, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_psnr", psnr_val, on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim", ssim_val, on_epoch=True, prog_bar=True)
        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, "test")

    # ---------- Optimizer ----------
    def configure_optimizers(self):
        backbone_params = [p for p in self.body.parameters() if p.requires_grad]
        optimizer = optim.Adam(
            [{"params": backbone_params,
              "lr": self.hparams.learning_rate * self.hparams.backbone_lr_scale}],
            weight_decay=1e-6,
        )
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