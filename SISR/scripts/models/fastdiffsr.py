"""
FastDiffSR x4 model definition
Original implementation: https://github.com/Meng-333/FastDiffSR | paper:10.1109/TGRS.2024.3458009

Expected inference input:
    LR_up: (B, 3, 256, 256)
            Three-channel LR image already upsampled x4.
Output:
    SR:    (B, 3, 256, 256)
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

# Project directories
SISR_ROOT = Path(__file__).resolve().parents[2]  # Superresolution-TIR/SISR
PROJECT_ROOT = SISR_ROOT.parent                  # Superresolution-TIR

# Allow imports from SISR/
sys.path.insert(0, str(SISR_ROOT))

# FastDiffSR source, configuration, and checkpoint
FASTDIFFSR_ROOT = SISR_ROOT / "scripts" / "external" / "FastDiffSR"
CONFIG_PATH = FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"

# A shared physical-temperature mapping for BOTH LR and HR.
# Celsius 0 -> -1; Celsius 70 -> +1.
# Keep these fixed between training, validation, and testing.
TEMP_MIN = 0.0
TEMP_MAX = 70.0


class FastDiffSRModule(pl.LightningModule):
    """
    Lightning adapter for pretrained FastDiffSR x4.

    Dataset tensors are independently standardized by SRDataset.
    We reverse that standardization, convert both LR and HR to a
    shared scale, and use FastDiffSR's native diffusion objective.
    """

    def __init__(
        self,
        stats_path,
        learning_rate=1e-5,
        weight_decay=1e-6,
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
    ):
        super().__init__()

        self.save_hyperparameters(
            ignore=["checkpoint_path", "config_path"]
        )

        # Dataset normalization statistics.
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.hr_mean = float(stats["hr"]["mean"])
        self.hr_std = float(stats["hr"]["std"])
        self.lr_mean = float(stats["lr"]["mean"])
        self.lr_std = float(stats["lr"]["std"])

        # Existing metrics helper expects these attributes.
        # These are the physical-temperature bounds used for
        # metric normalization, not the model's input normalization.
        self.DATA_MIN = float(TEMP_MIN)
        self.DATA_RANGE = float(TEMP_MAX - TEMP_MIN)

        # Import FastDiffSR using its original package structure.
        sys.path.insert(0, str(FASTDIFFSR_ROOT.parent))

        from FastDiffSR.fastdiffsr_modules import diffusion, unet

        with open(config_path, "r") as f:
            config = json.load(f)

        model_opt = config["model"]
        unet_opt = model_opt["unet"]
        diffusion_opt = model_opt["diffusion"]

        denoiser = unet.UNet(
            in_channel=unet_opt["in_channel"],
            out_channel=unet_opt["out_channel"],
            norm_groups=unet_opt.get("norm_groups", 32),
            inner_channel=unet_opt["inner_channel"],
            channel_mults=unet_opt["channel_multiplier"],
            attn_res=unet_opt["attn_res"],
            res_blocks=unet_opt["res_blocks"],
            dropout=unet_opt["dropout"],
            image_size=diffusion_opt["image_size"],
        )

        self.netG = diffusion.GaussianDiffusion(
            denoiser,
            image_size=diffusion_opt["image_size"],
            channels=diffusion_opt["channels"],
            loss_type="l1",
            conditional=diffusion_opt["conditional"],
            schedule_opt=None,
            scale=4,
        )

        # Initialize the diffusion schedule BEFORE loading the checkpoint.
        schedule = config["model"]["beta_schedule"]["train"]
        self.netG.set_new_noise_schedule(schedule, device=torch.device("cpu"))
        self.netG.set_loss(torch.device("cpu"))

        # Load pretrained FastDiffSR generator weights.
        checkpoint = torch.load(
            checkpoint_path,
            map_location="cpu",
            weights_only=True,
        )

        model_state = self.netG.state_dict()

        # These checkpoint parameters are absent from the official
        # GaussianDiffusion architecture and are not used in its
        # super_resolution inference path.
        extra_keys = {
            key for key in checkpoint
            if key.startswith("feature_extractor.")
        }

        usable_checkpoint = {
            key: value
            for key, value in checkpoint.items()
            if key not in extra_keys
        }

        # Keep strict validation for every parameter in the actual model.
        missing = set(model_state) - set(usable_checkpoint)
        unexpected = set(usable_checkpoint) - set(model_state)

        shape_errors = [
            (
                key,
                tuple(model_state[key].shape),
                tuple(usable_checkpoint[key].shape),
            )
            for key in model_state.keys() & usable_checkpoint.keys()
            if model_state[key].shape != usable_checkpoint[key].shape
        ]

        if missing or unexpected or shape_errors:
            raise RuntimeError(
                "Checkpoint mismatch:\n"
                f"Missing keys: {sorted(missing)}\n"
                f"Unexpected keys: {sorted(unexpected)}\n"
                f"Shape mismatches: {shape_errors}"
            )

        self.netG.load_state_dict(usable_checkpoint, strict=True)

        print(
            "[FastDiffSR] Checkpoint loaded: "
            f"{len(usable_checkpoint)} parameters; "
            f"excluded {len(extra_keys)} unused feature-extractor parameters."
        )


        # Values from the config make the expected geometry explicit.
        assert diffusion_opt["image_size"] == 256
        assert diffusion_opt["channels"] == 3
        assert unet_opt["in_channel"] == 6
        assert unet_opt["out_channel"] == 3

    # -----------------------------------------------------
    # Temperature scaling
    # -----------------------------------------------------

    @staticmethod
    def _to_physical(tensor, mean, std):
        """Undo SRDataset's standardization."""
        return tensor * std + mean

    @staticmethod
    def _to_model_scale(tensor):
        """Map Celsius to the shared [-1, 1] model scale."""
        return 2.0 * (tensor - TEMP_MIN) / (TEMP_MAX - TEMP_MIN) - 1.0

    @staticmethod
    def _to_celsius(tensor):
        """Undo the shared [-1, 1] model scale."""
        return (tensor + 1.0) * (TEMP_MAX - TEMP_MIN) / 2.0 + TEMP_MIN

    @staticmethod
    def _single_channel(tensor):
        """Use channel 0 because dataset channels are repeated copies."""
        return tensor[:, 0:1]

    def _prepare_batch(self, batch):
        # Recover physical temperatures from SRDataset normalization.
        lr_c = self._to_physical(
            self._single_channel(batch["lr"]),
            self.lr_mean,
            self.lr_std,
        )
        hr_c = self._to_physical(
            self._single_channel(batch["hr"]),
            self.hr_mean,
            self.hr_std,
        )

        # Both images must use the SAME mapping before computing residuals.
        lr = self._to_model_scale(lr_c)
        hr = self._to_model_scale(hr_c)

        # FastDiffSR conditions on the LR image at HR resolution.
        lr_up = F.interpolate(
            lr,
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        # Its UNet expects three conditioning channels plus three noisy
        # residual channels. The TIR image is single-channel, so repeat it.
        lr_up = lr_up.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)

        hr_mask = batch["hr_mask"].float()

        return lr_up, hr, hr_mask

    # -----------------------------------------------------
    # Masked diffusion training objective
    # -----------------------------------------------------

    def _masked_diffusion_loss(self, hr, lr_up, hr_mask):
        """
        FastDiffSR predicts the noise added to the HR residual.

        This follows p_losses() from the original implementation,
        but ignores invalid HR pixels in the loss.
        """
        diffusion = self.netG

        # Match the residual parameterization in the original code.
        x_start = diffusion.img2res(hr, lr_up)
        batch_size = x_start.shape[0]

        # Sample a diffusion timestep and continuous noise level.
        t = np.random.randint(1, diffusion.num_timesteps + 1)

        sqrt_alpha = torch.as_tensor(
            np.random.uniform(
                diffusion.sqrt_alphas_cumprod_prev[t - 1],
                diffusion.sqrt_alphas_cumprod_prev[t],
                size=batch_size,
            ),
            device=x_start.device,
            dtype=x_start.dtype,
        ).view(batch_size, 1, 1, 1)

        noise = torch.randn_like(x_start)

        x_noisy = diffusion.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=sqrt_alpha,
            noise=noise,
        )

        # Conditional FastDiffSR concatenates the LR condition and
        # noisy residual into the six-channel UNet input.
        noise_pred = diffusion.denoise_fn(
            torch.cat([lr_up, x_noisy], dim=1),
            sqrt_alpha.view(batch_size, -1),
        )

        # Repeat the HR validity mask for the three identical channels.
        mask = hr_mask.repeat(1, 3, 1, 1).to(noise_pred.dtype)

        abs_error = (noise_pred - noise).abs() * mask
        valid_count = mask.sum().clamp_min(1.0)

        return abs_error.sum() / valid_count

    # -----------------------------------------------------
    # Lightning steps
    # -----------------------------------------------------

    def forward(self, lr_up):
        if lr_up.shape[0] == 1:
            return self.netG.super_resolution(lr_up)

        predictions = [
            self.netG.super_resolution(lr_up[i:i + 1])
            for i in range(lr_up.shape[0])
        ]

        return torch.cat(predictions, dim=0)

    def training_step(self, batch, batch_idx):
        lr_up, hr, hr_mask = self._prepare_batch(batch)

        loss = self._masked_diffusion_loss(hr, lr_up, hr_mask)

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=hr.shape[0],
        )
        return loss

    def _evaluation_step(self, batch, stage):
        lr_up, hr_scaled, hr_mask = self._prepare_batch(batch)

        with torch.no_grad():
            sr_scaled = self(lr_up)

        # Convert predictions and reference temperatures back to Celsius.
        sr = self._to_celsius(sr_scaled[:, 0:1])
        hr = self._to_celsius(hr_scaled[:, 0:1])

        from scripts.utils.metrics import compute_metrics

        metrics_dict = compute_metrics(
            module=self,
            sr=sr,
            hr=hr,
            hr_mask=hr_mask,
            stage=stage,
            water_mask=batch.get("water_mask"),
        )

        # Log only metrics that are available.
        for name, value in metrics_dict.items():
            if value is None:
                continue

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)

            self.log(
                f"{stage}/{name}",
                value,
                on_step=False,
                on_epoch=True,
                prog_bar=(name == "water_mae"),
                batch_size=hr.shape[0],
            )

    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, "val")

    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, "test")

    # -----------------------------------------------------
    # Optimizer and scheduler
    # -----------------------------------------------------

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )

        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=0.5,
            patience=5,
        )

        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/water_mae",
                "interval": "epoch",
                "frequency": 1,
            },
        }