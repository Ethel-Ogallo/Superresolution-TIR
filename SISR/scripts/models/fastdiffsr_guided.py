"""
FastDiffSR x4 Lightning adapter for TIR SR.
Original implementation: https://github.com/Meng-333/FastDiffSR

Paper: 10.1109/TGRS.2024.3458009

Adaptation:
- Single-channel TIR, repeated to 3 channels to retain pretrained representation.
- 23 auxiliary channels + 2 temporal channels + 6 = 31-channel denoiser input.
- Custom sampling methods implemented in this adapter*
"""

from pathlib import Path
import json
import sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


# ------------- Paths and constants -----------------

SISR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SISR_ROOT.parent

sys.path.insert(0, str(SISR_ROOT))

FASTDIFFSR_ROOT = SISR_ROOT / "scripts" / "external" / "FastDiffSR"
CONFIG_PATH = FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"

sys.path.insert(0, str(FASTDIFFSR_ROOT.parent))

from FastDiffSR import networks


# ------------ Load pretrained weights ------------

def load_pretrained_weights(netG, checkpoint_path):
    """
    Load the original FastDiffSR checkpoint.
    The original six-channel architecture must be loaded before expanding its first convolution to 
    accommodate auxiliary data.
    """

    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # Exclude unused feature-extractor parameters, if present.
    extra_keys = {k for k in checkpoint if k.startswith("feature_extractor.")}
    weights = {k: v for k, v in checkpoint.items() if k not in extra_keys}

    model_state = netG.state_dict()

    missing = set(model_state) - set(weights)
    unexpected = set(weights) - set(model_state)

    shape_errors = [
        (k,tuple(model_state[k].shape),tuple(weights[k].shape),)
        for k in model_state.keys() & weights.keys() if model_state[k].shape != weights[k].shape
    ]

    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Checkpoint mismatch:\n"
            f"Missing keys: {sorted(missing)}\n"
            f"Unexpected keys: {sorted(unexpected)}\n"
            f"Shape mismatches: {shape_errors}" )

    netG.load_state_dict(weights, strict=True)

    print(
        f"[FastDiffSR] Checkpoint loaded: {len(weights)} parameters; "
        # f"excluded {len(extra_keys)} unused feature-extractor parameters."
    )


# ---------------- expand the first UNet convolution ------------
def expand_input_conv(conv, new_in_channels):
    """
    Expand the input convolution while preserving pretrained weights.

    Original input: 6 channels = 3 LR + 3 noisy residual
    New input: 31 channels = 3 LR + 3 noisy residual + 25 auxiliary/time

    The additional 25 channels are initialized to zero.
    """

    new_conv = nn.Conv2d(
        in_channels  = new_in_channels,
        out_channels = conv.out_channels,
        kernel_size  = conv.kernel_size,
        stride       = conv.stride,
        padding      = conv.padding,
        dilation     = conv.dilation,
        groups       = conv.groups,
        bias         = conv.bias is not None,
        padding_mode = conv.padding_mode,
    )

    # Preserve the pretrained weights and initialize new channels to zero.
    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


# ---------------- FastDiffSR Lightning Module ---------------

class FastDiffSRModule(pl.LightningModule):

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

        # Dataset statistics
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.hr_mean = float(stats["hr"]["mean"])
        self.hr_std = float(stats["hr"]["std"])
        self.lr_mean = float(stats["lr"]["mean"])
        self.lr_std = float(stats["lr"]["std"])
        self.temp_min = float(min(stats["hr"]["min"], stats["lr"]["min"]))
        self.temp_max = float(max(stats["hr"]["max"], stats["lr"]["max"]))
        self.DATA_MIN = float(stats["hr_percentiles"]["p1"])
        self.DATA_RANGE = float(stats["hr_data_range"])

        # Load original FastDiffSR architecture
        with open(config_path, "r") as f:
            config = json.load(f)

        config["distributed"] = False

        self.netG = networks.define_G(config)

        # Set the original diffusion noise schedule.
        self.netG.set_new_noise_schedule(
            config["model"]["beta_schedule"]["train"],device=torch.device("cpu"),)

        self.netG.set_loss(torch.device("cpu"))

        # Load pretrained weights.
        load_pretrained_weights( self.netG, checkpoint_path,)

        # Expand UNet input convolution: 6 -> 31 channels
        first_conv = self.netG.denoise_fn.downs[0]
        self.netG.denoise_fn.downs[0] = expand_input_conv(
            first_conv,
            new_in_channels=31,
        )

        print("[FastDiffSR] UNet input expanded from 6 to 31 channels.")

    # Temperature scaling
    @staticmethod
    def _single_channel(tensor):
        """Keep one copy of the single-channel TIR image."""
        return tensor[:, 0:1]

    @staticmethod
    def _to_physical(tensor, mean, std):
        """Convert z-score standardized values back to Celsius."""
        return tensor * std + mean

    def _to_model_scale(self, tensor):
        """Map Celsius values to [-1, 1]."""
        return ( 2.0 * (tensor - self.temp_min)/ (self.temp_max - self.temp_min) - 1.0)

    def _to_celsius(self, tensor):
        """Convert model-scale values back to Celsius."""
        return ((tensor + 1.0) * (self.temp_max - self.temp_min)/ 2.0 + self.temp_min)

    # Batch preparation
    def _prepare_batch(self, batch):
        # Convert standardized LR and HR to Celsius.
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

        # Map both images to the model's [-1, 1] range.
        lr = self._to_model_scale(lr_c)
        hr = self._to_model_scale(hr_c)

        # Upsample LR to the HR spatial resolution.
        lr_up = F.interpolate(
            lr,
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        )

        # Retain FastDiffSR's original three-channel representation.
        lr_up = lr_up.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)

        # Auxiliary data
        aux = batch["aux"].float()
        time_channels = batch["time_channels"].float()
        aux = torch.cat([aux, time_channels],dim=1,)

        # Check expected dimensions.
        assert aux.shape[1] == 25, (f"Expected 25 auxiliary channels, got {aux.shape[1]}")
        assert aux.shape[-2:] == hr.shape[-2:], (
            f"Auxiliary resolution {aux.shape[-2:]} " f"does not match HR resolution {hr.shape[-2:]}")
        assert aux.shape[0] == hr.shape[0]

        return (lr_up, hr, batch["hr_mask"].float(), aux, )

    # Training: masked diffusion loss
    def _masked_diffusion_loss(self, hr, lr_up, hr_mask, aux):
        diffusion = self.netG

        # Calculate HR residual relative to bicubic-upsampled LR.
        x_start = diffusion.img2res(hr, lr_up)
        batch_size = x_start.shape[0]
        
        t = np.random.randint(1, diffusion.num_timesteps + 1) # Sample diffusion timestep.

        # Sample continuous noise strength independently per image.
        sqrt_alpha = torch.as_tensor(
            np.random.uniform(
                diffusion.sqrt_alphas_cumprod_prev[t - 1],
                diffusion.sqrt_alphas_cumprod_prev[t],
                size=batch_size,
            ),
            device=x_start.device,
            dtype=x_start.dtype,
        ).view(batch_size, 1, 1, 1)

        # Generate Gaussian noise and corrupt the clean residual.
        noise = torch.randn_like(x_start)

        x_noisy = diffusion.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=sqrt_alpha,
            noise=noise,
        )

        # Concatenate LR, noisy residual and auxiliary data.
        denoiser_input = torch.cat(
            [lr_up, x_noisy, aux],
            dim=1,
        )

        # Predict the noise.
        noise_pred = diffusion.denoise_fn(
            denoiser_input,
            sqrt_alpha.view(batch_size, -1),
        )

        # Calculate masked diffusion noise-prediction L1 loss.
        mask = hr_mask.repeat(1, 3, 1, 1).to(noise_pred.dtype)
        abs_error = (noise_pred - noise).abs() * mask

        return abs_error.sum() / mask.sum().clamp_min(1.0)


    # Custom diffusion sampling methods
    # These replace the original methods for inference only.
    def _p_mean_variance(self,x, t,condition_x, aux,clip_denoised=True,):
        """
        Calculate the posterior mean and log variance.
        x:Current noisy residual (3 channels).
        condition_x: Bicubic-upsampled LR TIR (3 channels).
        aux: Auxiliary and temporal data (25 channels).
        """

        diffusion = self.netG
        batch_size = x.shape[0]

        # Noise-level embedding used by the original FastDiffSR UNet.
        noise_level = torch.full(
            (batch_size, 1),
            float(diffusion.sqrt_alphas_cumprod_prev[t + 1]),
            device=x.device,
            dtype=x.dtype,
        )

        # The auxiliary channels are added only at the denoiser.
        denoiser_input = torch.cat([condition_x, x, aux],dim=1,)

        noise_pred = diffusion.denoise_fn(denoiser_input,noise_level,)

        # Predict the clean residual from the noisy residual.
        x_recon = diffusion.predict_start_from_noise( x,t=t, noise=noise_pred, )
        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        # Calculate the diffusion posterior.
        model_mean, model_log_variance = diffusion.q_posterior(x_start=x_recon, x_t=x, t=t,)

        return model_mean, model_log_variance

    @torch.no_grad()
    def _p_sample(self, x, t,condition_x, aux,clip_denoised=True,):
        """
        Sample the residual at the previous diffusion timestep.
        """
        model_mean, model_log_variance = self._p_mean_variance(
            x=x,t=t,condition_x=condition_x,aux=aux, clip_denoised=False,)

        # Add random noise except at the final timestep.
        noise = (
            torch.randn_like(x)
            if t > 0
            else torch.zeros_like(x)
        )

        return ( model_mean + noise * (0.5 * model_log_variance).exp() )

    @torch.no_grad()
    def _super_resolution(self, lr_up, aux):
        """
        Custom FastDiffSR sampling loop with auxiliary conditioning.

        The diffusion process operates on a three-channel residual.
        Auxiliary data is passed separately to the denoiser.
        """

        diffusion = self.netG

        # Initialize the three-channel residual with Gaussian noise.
        img = torch.randn_like(lr_up)

        # Iteratively denoise the residual.
        for t in reversed(range(diffusion.num_timesteps)):

            img = self._p_sample(x=img,t=t,condition_x=lr_up, aux=aux, )

        # Convert the final residual back into an SR image.
        sr = diffusion.res2img( img, lr_up,)

        return sr

    # Forward pass
    def forward(self, lr_up, aux):

        outputs = []

        for i in range(lr_up.shape[0]):
            sr = self._super_resolution( lr_up[i:i + 1], aux[i:i + 1],)
            outputs.append(sr)

        return torch.cat(outputs, dim=0)

    # Training step
    def training_step(self, batch, batch_idx):
        lr_up, hr, hr_mask, aux = self._prepare_batch(batch)
        loss = self._masked_diffusion_loss( hr,lr_up, hr_mask, aux, )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=hr.shape[0],
        )

        return loss

    # Validation and testing steps
    def _evaluation_step(self, batch, stage):

        lr_up, hr_scaled, hr_mask, aux = self._prepare_batch(batch)

        # Generate SR predictions.
        with torch.no_grad():
            sr_scaled = self(lr_up, aux)

        # Convert predictions and HR to Celsius.
        sr = self._to_celsius(sr_scaled[:, 0:1])
        hr = self._to_celsius(hr_scaled[:, 0:1])

        from scripts.utils.metrics import compute_metrics

        metrics = compute_metrics(
            module=self,
            sr=sr,
            hr=hr,
            hr_mask=hr_mask,
            stage=stage,
            water_mask=batch.get("water_mask"),
        )

        for name, value in metrics.items():
            if value is None:
                continue

            if not isinstance(value, torch.Tensor):
                value = torch.tensor(
                    value,
                    device=self.device,
                    dtype=torch.float32,
                )

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

    # Optimizer and scheduler
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