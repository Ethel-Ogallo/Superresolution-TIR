# scripts/models/resshift.py
"""
ResShift Lightning module for TIR Super-Resolution.

Phase 1: zero-shot inference using official ResShift sampler
Phase 2: fine-tuning using official ResShift diffusion training

Key design:
- sampler.model   = the actual UNetModelSwin (properly loaded)
- sampler.autoencoder = VQ autoencoder
- sampler.base_diffusion = GaussianDiffusion with ResShift schedule
- self.body = unused dummy (required by Lightning but not trained)

Both phases use sampler.model — consistent and truthful.
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.optim as optim
import lightning.pytorch as pl
from pathlib import Path

RESSHIFT_ROOT = Path(__file__).resolve().parents[1] / "external" / "resshift"
if str(RESSHIFT_ROOT) not in sys.path:
    sys.path.insert(0, str(RESSHIFT_ROOT))

from scripts.utils.metrics import compute_metrics
from scripts.utils.loss import masked_l1


class ResShiftModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path: str  = None,
        ae_path: str          = None,
        hr_mean: float        = 0.0,
        hr_std: float         = 1.0,
        data_range: float     = 70.0,
        learning_rate: float  = 1e-4,
        num_steps: int        = 15,
        scale: int            = 4,
        phase: int            = 1,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.DATA_RANGE      = float(data_range)
        self._sampler        = None
        self.pretrained_path = pretrained_path
        self.ae_path         = ae_path

        # Dummy parameter so Lightning is happy
        # Actual model is sampler.model loaded via official checkpoint
        self._dummy = torch.nn.Parameter(torch.zeros(1))

    # ── Sampler — lazy init ───────────────────────────────────────────────────
    def _get_sampler(self):
        if self._sampler is not None:
            return self._sampler

        if self.pretrained_path is None or not os.path.exists(str(self.pretrained_path)):
            raise ValueError(
                f"pretrained_path is required for ResShift sampler. "
                f"Got: {self.pretrained_path}"
            )
        if self.ae_path is None or not os.path.exists(str(self.ae_path)):
            raise ValueError(
                f"ae_path is required for ResShift sampler. "
                f"Got: {self.ae_path}"
            )

        from unittest.mock import MagicMock
        sys.modules['datapipe']          = MagicMock()
        sys.modules['datapipe.datasets'] = MagicMock()

        os.environ.setdefault('LOCAL_RANK', '0')

        from omegaconf import OmegaConf
        from sampler import ResShiftSampler

        cfg_path = RESSHIFT_ROOT / "configs" / \
                   "realsr_swinunet_realesrgan256.yaml"
        cfg      = OmegaConf.load(str(cfg_path))

        cfg.model.ckpt_path       = str(self.pretrained_path)
        cfg.autoencoder.ckpt_path = str(self.ae_path)

        self._sampler = ResShiftSampler(
            configs  = cfg,
            sf       = self.hparams.scale,
            use_fp16 = False,
        )
        print("[INFO] ResShift sampler initialised")
        return self._sampler

    # ── Denormalise ───────────────────────────────────────────────────────────
    def denormalize(self, t):
        mean = torch.tensor(self.hparams.hr_mean, device=t.device)
        std  = torch.tensor(self.hparams.hr_std,  device=t.device)
        return t * std + mean

    # ── Normalise tile to [0,1] for ResShift ─────────────────────────────────
    def _to_resshift_range(self, x_celsius):
        """
        Convert Celsius tensor to [0,1] per-image range for ResShift.
        Returns (x_01, t_min, t_range) for inverse mapping.
        """
        t_min   = x_celsius.flatten(1).min(dim=1)[0][:, None, None, None]
        t_max   = x_celsius.flatten(1).max(dim=1)[0][:, None, None, None]
        t_range = (t_max - t_min).clamp(min=1e-6)
        x_01    = (x_celsius - t_min) / t_range
        return x_01, t_min, t_range

    # ── Phase 1 sampling ──────────────────────────────────────────────────────
    @torch.no_grad()
    def _sample(self, lr):
        """
        Full ResShift diffusion sampling using official sampler.
        lr: (B, 3, 64, 64) normalised tensor
        returns: (B, 3, 256, 256) normalised SR tensor
        """
        sampler = self._get_sampler()

        lr_celsius          = lr[:, 0:1] * self.hparams.hr_std \
                            + self.hparams.hr_mean
        lr_01, t_min, t_range = self._to_resshift_range(lr_celsius)
        lr_3ch              = lr_01.repeat(1, 3, 1, 1)

        # sample_func expects [0,1] — it converts to [-1,1] internally
        sr_01 = sampler.sample_func(
            y0           = lr_3ch,
            noise_repeat = False,
        )                                    # (B, 3, 256, 256) in [-1,1]

        # sampler returns [-1,1] — convert to [0,1]
        sr_01 = (sr_01 + 1) / 2

        # Map back to Celsius using LR tile stats
        sr_celsius = sr_01[:, 0:1] * t_range + t_min
        sr_norm    = (sr_celsius - self.hparams.hr_mean) / self.hparams.hr_std
        return sr_norm.repeat(1, 3, 1, 1)

    # ── Phase 2 diffusion training ────────────────────────────────────────────
    def _diffusion_training_step(self, batch):
        """
        Proper ResShift diffusion training using official training_losses API.

        GaussianDiffusion.training_losses(model, x_start, y, t,
                                          first_stage_model, model_kwargs)
        where:
            x_start = HR image in [0,1]
            y       = LR upsampled to HR size in [0,1]
            t       = random timestep
        """
        lr_img  = batch["lr"]       # (B, 3, 64,  64) normalised
        hr_img  = batch["hr"]       # (B, 3, 256, 256) normalised
        hr_mask = batch["hr_mask"]  # (B, 1, 256, 256)

        sampler = self._get_sampler()
        b       = hr_img.shape[0]
        device  = hr_img.device

        # Convert HR to Celsius then [0,1]
        hr_celsius          = hr_img[:, 0:1] * self.hparams.hr_std \
                            + self.hparams.hr_mean
        hr_01, t_min, t_range = self._to_resshift_range(hr_celsius)
        hr_01               = hr_01.repeat(1, 3, 1, 1)

        # Convert LR to Celsius then [0,1] using same stats as HR
        lr_celsius = lr_img[:, 0:1] * self.hparams.hr_std \
                   + self.hparams.hr_mean
        lr_01      = ((lr_celsius - t_min) / t_range).clamp(0, 1)
        lr_01      = lr_01.repeat(1, 3, 1, 1)

        # Upsample LR to HR size for conditioning
        lr_up = F.interpolate(
            lr_01, size=hr_01.shape[-2:],
            mode="bicubic", align_corners=False
        )

        # Random timestep
        t = torch.randint(
            0, sampler.base_diffusion.num_timesteps,
            (b,), device=device
        )

        # Official ResShift training loss
        losses = sampler.base_diffusion.training_losses(
            model             = sampler.model,
            x_start           = hr_01,
            y                 = lr_up,
            t                 = t,
            first_stage_model = sampler.autoencoder,
            model_kwargs      = {"lq": lr_up},
        )

        # losses["loss"] shape: (B,) — mean over batch
        loss = losses["loss"].mean()
        return loss

    # ── Steps ─────────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        loss = self._diffusion_training_step(batch)
        self.log("train_loss", loss, on_step=True,
                 on_epoch=True, prog_bar=True)
        return loss

    def validation_step(self, batch, batch_idx):
        """Fast validation — use sample_func for honest diffusion eval."""
        lr_img     = batch["lr"]
        hr_img     = batch["hr"]
        hr_mask    = batch["hr_mask"]
        water_mask = batch.get("water_mask", None)

        with torch.no_grad():
            sr_img = self._sample(lr_img)

        sr = self.denormalize(sr_img[:, 0:1])
        hr = self.denormalize(hr_img[:, 0:1])

        loss = masked_l1(sr, hr, hr_mask)
        self.log("val_loss", loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        with torch.no_grad():
            compute_metrics(self, sr, hr, hr_mask, "val", water_mask)

        return loss

    def test_step(self, batch, batch_idx):
        """Full diffusion sampling for honest test evaluation."""
        lr_img     = batch["lr"]
        hr_img     = batch["hr"]
        hr_mask    = batch["hr_mask"]
        water_mask = batch.get("water_mask", None)

        with torch.no_grad():
            sr_img = self._sample(lr_img)

        sr = self.denormalize(sr_img[:, 0:1])
        hr = self.denormalize(hr_img[:, 0:1])

        loss = masked_l1(sr, hr, hr_mask)
        self.log("test_loss", loss, on_epoch=True,
                 prog_bar=True, sync_dist=True)

        with torch.no_grad():
            compute_metrics(self, sr, hr, hr_mask, "test", water_mask)

        return loss

    # ── Optimiser — optimise sampler.model for Phase 2 ───────────────────────
    def configure_optimizers(self):
        sampler = self._get_sampler()
        params  = sampler.model.parameters()

        opt = optim.Adam(
            params,
            lr           = self.hparams.learning_rate,
            weight_decay = 1e-6
        )
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=5
        )
        return {
            "optimizer":    opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_psnr"}
        }

    # ── Not used for training but kept for Lightning ckpt compatibility ───────
    def forward(self, lr):
        return self._sample(lr)