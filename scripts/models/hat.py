"""
hat.py — HAT Lightning module for TIR super-resolution (x4).
Based on the HAT architecture from XPixelGroup/HAT, with a PyTorch Lightning wrapper for training and evaluation.
"""

import os
import sys
import types
import requests  # noqa: PLC0415
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from torchmetrics.image import PeakSignalNoiseRatio, StructuralSimilarityIndexMeasure

from scripts.utils.loss import gradient_loss

# ----------------- Arch import ---------------  
def _load_hat_arch():
    """Load the HAT architecture, either from basicsr or GitHub."""
    try:
        from basicsr.archs import hat_arch as _hat_arch
        print("[INFO] HAT arch loaded from basicsr.archs")
        return _hat_arch
    except ImportError:
        pass

    if "hat_arch" in sys.modules:
        return sys.modules["hat_arch"]

    print("[INFO] basicsr HAT not found — fetching hat_arch from XPixelGroup/HAT on GitHub...")

    _HAT_URL = (
        "https://raw.githubusercontent.com/XPixelGroup/HAT/main/hat/archs/hat_arch.py"
    )
    resp = requests.get(_HAT_URL, timeout=30)
    resp.raise_for_status()
    _module = types.ModuleType("hat_arch")
    exec(resp.text, _module.__dict__)  # noqa: S102
    sys.modules["hat_arch"] = _module
    print("[INFO] HAT arch loaded from GitHub and cached in sys.modules")
    return _module


_hat_arch = _load_hat_arch()
HAT = _hat_arch.HAT  # The top-level HAT class (upscale, in_chans, img_size, …)


#  --------------- model def ------------------                                                              
class HATModule(pl.LightningModule):
    """
    Lightning wrapper for HAT super-resolution.
    """

    # Head layer names used to split backbone vs head param groups
    _HEAD_KEYWORDS = ["conv_last", "upsample", "conv_before_upsample", "conv_hr"]

    def __init__(
        self,
        pretrained_path: str = None,
        mean: float = 0.0,
        std: float = 1.0,
        learning_rate: float = 1e-4,
        bb_lr_scale: float = 0.1,
        patience: int = 15,
        img_size: int = 64,
        embed_dim: int = 180,
        depths: list = None,
        num_heads: list = None,
        window_size: int = 16,
        mlp_ratio: float = 2.0,
        compress_ratio: int = 3,
        squeeze_factor: int = 30,
        overlap_ratio: float = 0.5,
        upsampler: str = "pixelshuffle",
        freeze_backbone: bool = False,
        lambda_grad: float = 0.1,
        data_range: float = 80.0,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.DATA_RANGE = float(data_range)
        print(f"[INFO] DATA_RANGE set to {self.DATA_RANGE}°C")

        depths    = depths    or [6, 6, 6, 6, 6, 6]
        num_heads = num_heads or [6, 6, 6, 6, 6, 6]

        # Build backbone 
        self.body = HAT(
            upscale        = 4,
            in_chans       = 3,
            img_size       = img_size,
            window_size    = window_size,
            compress_ratio = compress_ratio,
            squeeze_factor = squeeze_factor,
            conv_scale     = 0.01,
            overlap_ratio  = overlap_ratio,
            img_range      = 1.0,
            depths         = depths,
            embed_dim      = embed_dim,
            num_heads      = num_heads,
            mlp_ratio      = mlp_ratio,
            upsampler      = upsampler,
            resi_connection= "1conv",
        )

        # Load pretrained 
        if pretrained_path and os.path.exists(pretrained_path):
            self._load_pretrained(pretrained_path)
        else:
            print("[INFO] No pretrained weights; training from scratch")

        # Now replace the very last Conv2d with a 1-ch equivalent
        self._replaced_conv_name = self._replace_last_conv(out_channels=1)

        self._set_backbone_frozen(freeze_backbone)

        for split in ("train", "val", "test"):
            setattr(self, f"{split}_psnr",
                    PeakSignalNoiseRatio(data_range=self.DATA_RANGE))
            setattr(self, f"{split}_ssim",
                    StructuralSimilarityIndexMeasure(data_range=self.DATA_RANGE))

    # --------------- Helpers  --------------------------                                                        
    def _replace_last_conv(self, out_channels: int = 1) -> str | None:
        """
        Walk self.body, find the last Conv2d, replace it with a single-output
        conv of the same kernel/padding.  Returns the dotted layer name.
        """
        last_conv_name = None
        for name, module in self.body.named_modules():
            if isinstance(module, nn.Conv2d):
                last_conv_name = name

        if last_conv_name is None:
            print("[ERROR] Could not find any Conv2d in body ")
            return None

        parts  = last_conv_name.split(".")
        parent = self.body
        for part in parts[:-1]:
            parent = getattr(parent, part)

        old_conv = getattr(parent, parts[-1])
        new_conv = nn.Conv2d(
            old_conv.in_channels,
            out_channels,
            kernel_size=old_conv.kernel_size,
            padding=old_conv.padding,
        )
        setattr(parent, parts[-1], new_conv)
        print(
            f"[INFO] Replaced {last_conv_name}: "
            f"{old_conv.out_channels}ch to {out_channels}ch"
        )
        return last_conv_name

    def denormalize(self, t: torch.Tensor) -> torch.Tensor:
        mean = torch.tensor(self.hparams.mean, device=t.device)
        std  = torch.tensor(self.hparams.std,  device=t.device)
        return t * std + mean

    @staticmethod
    def crop_to_valid_bbox(tensor: torch.Tensor, mask: torch.Tensor):
        """Crop to the tight bounding box of valid (non-zero) mask pixels."""
        m    = mask[:, 0, :, :]
        flat = m.any(dim=0)
        rows = flat.any(dim=1).nonzero(as_tuple=True)[0]
        cols = flat.any(dim=0).nonzero(as_tuple=True)[0]
        if rows.numel() == 0 or cols.numel() == 0:
            return tensor, mask
        r0, r1 = rows[0].item(), rows[-1].item() + 1
        c0, c1 = cols[0].item(), cols[-1].item() + 1
        return tensor[:, :, r0:r1, c0:c1], mask[:, :, r0:r1, c0:c1]

    #  Pretrained loading                                              
    def _load_pretrained(self, path: str):
        print(f"[INFO] Loading pretrained HAT weights: {path}")
        ckpt       = torch.load(path, map_location="cpu", weights_only=False)
        state_dict = ckpt.get("params_ema", ckpt.get("params", ckpt))

        # Skip output / relative-position-index layers which are architecture-specific and likely to differ between pretrained and current model
        skip_keys  = {"conv_last", "upsample", "conv_before_upsample", "relative_position_index"}
        model_dict = self.body.state_dict()

        matched = {
            k: v
            for k, v in state_dict.items()
            if k in model_dict
            and v.shape == model_dict[k].shape
            and not any(s in k for s in skip_keys)
        }

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] Loaded {len(matched)}/{len(model_dict)} layers")

    #  Freeze / unfreeze logic                                              
    def _set_backbone_frozen(self, frozen: bool):
        """
        frozen=True  → only head layers are trainable (fast fine-tune)
        frozen=False → all parameters are trainable
        """
        for name, param in self.body.named_parameters():
            is_head = any(k in name for k in self._HEAD_KEYWORDS)
            param.requires_grad = is_head if frozen else True

    # def freeze_backbone(self):
    #     """Public helper — call from a callback or interactively."""
    #     self._set_backbone_frozen(True)
    #     print("[INFO] Backbone frozen — only head layers will be updated")

    # def unfreeze_backbone(self):
    #     """Public helper — call from a callback or interactively."""
    #     self._set_backbone_frozen(False)
    #     print("[INFO] Backbone unfrozen — all layers will be updated")

    # Forward                        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x3 = x.repeat(1, 3, 1, 1)
        sr = self.body(x3)
        # Safety: ensure 1-channel output always
        if sr.shape[1] != 1:
            sr = sr[:, :1, :, :]
        return sr

    # ------------------- Shared step (train / val / test)  -------------------                               
    def _shared_step(self, batch, batch_idx: int, stage: str):
        lr_img, hr_img, hr_mask = batch
        sr_img = self(lr_img)

        sr = self.denormalize(sr_img)
        hr = self.denormalize(hr_img)

        # Guard against NaNs / Infs propagating into loss
        sr = torch.nan_to_num(sr, nan=0.0, posinf=1e6, neginf=-1e6)
        hr = torch.nan_to_num(hr, nan=0.0, posinf=1e6, neginf=-1e6)

        # Masked L1 reconstruction loss (only valid pixels contribute)
        abs_err    = torch.abs(sr - hr) * hr_mask
        recon_loss = abs_err.sum() / torch.clamp(hr_mask.sum(), min=1.0)

        # Gradient loss 
        if self.hparams.lambda_grad > 0:
            grad_loss = gradient_loss(sr, hr, hr_mask)
            loss      = recon_loss + self.hparams.lambda_grad * grad_loss
        else:
            grad_loss = torch.tensor(0.0, device=sr.device)
            loss      = recon_loss

        # Final NaN guard on losses
        recon_loss = torch.nan_to_num(recon_loss, nan=0.0)
        grad_loss  = torch.nan_to_num(grad_loss,  nan=0.0)
        loss       = torch.nan_to_num(loss,        nan=0.0)

        # --------------- Metrics -------------------
        with torch.no_grad():
            sr_crop, mask_crop = self.crop_to_valid_bbox(sr, hr_mask)
            hr_crop, _         = self.crop_to_valid_bbox(hr, hr_mask)
            valid_mask         = (mask_crop > 0.5)  # (B, 1, H, W)

            if valid_mask.sum() == 0:
                psnr_val = torch.tensor(0.0, device=sr.device)
                ssim_val = torch.tensor(0.0, device=sr.device)
                mae_val  = torch.tensor(0.0, device=sr.device)
            else:
                mae_val = (sr_crop - hr_crop).abs()[valid_mask].mean()

                # Zero out invalid pixels so they don't skew PSNR/SSIM
                sr_for_metric = sr_crop.clone()
                hr_for_metric = hr_crop.clone()
                sr_for_metric[~valid_mask] = hr_for_metric[~valid_mask]

                psnr_val = getattr(self, f"{stage}_psnr")(
                    sr_for_metric, hr_for_metric
                )

                h, w = sr_crop.shape[-2:]
                if h < 11 or w < 11:
                    # SSIM uses an 11×11 window — skip for tiny crops
                    ssim_val = torch.tensor(0.0, device=sr.device)
                else:
                    ssim_val = getattr(self, f"{stage}_ssim")(
                        sr_for_metric, hr_for_metric
                    )

        # --------- Logging ------------------
        self.log(f"{stage}_loss",       loss,       on_epoch=True, prog_bar=True)
        self.log(f"{stage}_recon_loss", recon_loss, on_epoch=True)
        self.log(f"{stage}_grad_loss",  grad_loss,  on_epoch=True)
        self.log(f"{stage}_psnr",       psnr_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_ssim",       ssim_val,   on_epoch=True, prog_bar=True)
        self.log(f"{stage}_mae",        mae_val,    on_epoch=True, prog_bar=True)

        return loss

    def training_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "train")

    def validation_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "val")

    def test_step(self, batch, batch_idx):
        return self._shared_step(batch, batch_idx, "test")

    #  -----------------  Optimiser -----------------
    def configure_optimizers(self):
        backbone_params, head_params = [], []

        for name, p in self.body.named_parameters():
            if not p.requires_grad:
                continue
            if any(k in name for k in self._HEAD_KEYWORDS):
                head_params.append(p)
            else:
                backbone_params.append(p)

        optimizer = optim.Adam(
            [
                {"params": backbone_params,
                 "lr": self.hparams.learning_rate * self.hparams.bb_lr_scale},
                {"params": head_params,
                 "lr": self.hparams.learning_rate},
            ],
            weight_decay=1e-6,
        )

        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, 
            mode="max", 
            factor=0.5, 
            patience=self.hparams.patience
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_psnr"},
        }