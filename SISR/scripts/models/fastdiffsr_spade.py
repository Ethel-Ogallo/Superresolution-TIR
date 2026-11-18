"""
FastDiffSR x4 Lightning adapter for TIR super-resolution.

Denoiser input:
    3 LR channels + 3 noisy-residual channels + 23 auxiliary channels
    + 2 temporal channels = 31 channels.
"""

from pathlib import Path
import json
import sys

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import lightning.pytorch as pl


# ------------ Paths and imports -------------
SISR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SISR_ROOT.parent
FASTDIFFSR_ROOT = SISR_ROOT / "scripts" / "external" / "FastDiffSR"

sys.path.insert(0, str(SISR_ROOT))
sys.path.insert(0, str(FASTDIFFSR_ROOT.parent))

from scripts.utils.custom_loss import combined_loss
from scripts.utils.metrics import compute_metrics
from FastDiffSR import networks
from scripts.models.spade_diff import SPADEWrapper

CONFIG_PATH = FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"

# --------- Pretrained weights and input convolution ------------
def load_pretrained_weights(netG, checkpoint_path):
    """Load the original checkpoint before expanding the denoiser input."""
    checkpoint = torch.load(
        checkpoint_path,
        map_location="cpu",
        weights_only=True,
    )

    # The feature extractor is not part of the adapted generator.
    weights = {
        key: value
        for key, value in checkpoint.items()
        if not key.startswith("feature_extractor.")
        }

    model_state = netG.state_dict()

    missing = set(model_state) - set(weights)
    unexpected = set(weights) - set(model_state)
    mismatched = [
        (key, tuple(model_state[key].shape), tuple(weights[key].shape))
        for key in model_state.keys() & weights.keys()
        if model_state[key].shape != weights[key].shape
    ]

    if missing or unexpected or mismatched:
        raise RuntimeError(
            f"Checkpoint mismatch.\n"
            f"Missing: {sorted(missing)}\n"
            f"Unexpected: {sorted(unexpected)}\n"
            f"Shape mismatches: {mismatched}"
        )

    netG.load_state_dict(weights, strict=True)
    print(f"[FastDiffSR] Loaded {len(weights)} pretrained parameters.")

def expand_input_conv(conv, new_in_channels=31):
    """Preserve the original six channels and zero-initialize added channels."""
    new_conv = nn.Conv2d(
        in_channels=new_in_channels,
        out_channels=conv.out_channels,
        kernel_size=conv.kernel_size,
        stride=conv.stride,
        padding=conv.padding,
        dilation=conv.dilation,
        groups=conv.groups,
        bias=conv.bias is not None,
        padding_mode=conv.padding_mode,
        device=conv.weight.device,
        dtype=conv.weight.dtype,
    )

    with torch.no_grad():
        new_conv.weight.zero_()
        new_conv.weight[:, :conv.in_channels].copy_(conv.weight)
        if conv.bias is not None:
            new_conv.bias.copy_(conv.bias)

    return new_conv


# --------- FastDiffSR Lightning module ---------
class FastDiffSRModule(pl.LightningModule):
    def __init__(
        self,
        stats_path,
        learning_rate=1e-5,
        weight_decay=1e-6,
        checkpoint_path=CHECKPOINT_PATH,
        config_path=CONFIG_PATH,
        lambda_w=2.0,
        lambda_nw=1.0,
        lambda_g=0.1,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["checkpoint_path", "config_path"])

        # Dataset statistics
        with open(stats_path, "r") as f:
            stats = json.load(f)

        self.hr_mean = float(stats["hr"]["mean"])
        self.hr_std = float(stats["hr"]["std"])
        self.lr_mean = float(stats["lr"]["mean"])
        self.lr_std = float(stats["lr"]["std"])
        self.temp_min = float(min(stats["hr"]["min"], stats["lr"]["min"]))
        self.temp_max = float(max(stats["hr"]["max"], stats["lr"]["max"]))
        self.DATA_RANGE = float(stats["hr_data_range"])
        self.DATA_MIN = float(stats["hr_percentiles"]["p1"])
        self.lambda_w = lambda_w
        self.lambda_nw = lambda_nw
        self.lambda_g = lambda_g

        # Build the original generator and diffusion schedule.
        with open(config_path, "r") as f:
            config = json.load(f)

        config["distributed"] = False
        self.netG = networks.define_G(config)

        self.netG.set_new_noise_schedule(
            config["model"]["beta_schedule"]["train"],
            device=torch.device("cpu"),
        )
        self.netG.set_loss(torch.device("cpu"))

        # Load pretrained weights before expanding the first convolution.
        load_pretrained_weights(self.netG, checkpoint_path)

        first_conv = self.netG.denoise_fn.downs[0]
        self.netG.denoise_fn.downs[0] = expand_input_conv(first_conv, 31)

        print("[FastDiffSR] Denoiser input expanded from 6 to 31 channels.")

        # Wrap the pretrained denoiser with multiscale SPADE conditioning.
        self.netG.denoise_fn = SPADEWrapper(
            unet=self.netG.denoise_fn,
            aux_channels=23,
            label_nc=14,
            label_start=8,
        )

        print("[FastDiffSR] SPADE conditioning initialized.")


    # --------- Temperature scaling and batch preparation ---------
    @staticmethod
    def _to_physical(tensor, mean, std):
        """Convert dataset z-score values back to Celsius."""
        return tensor * std + mean

    def _to_model_scale(self, tensor):
        """Map Celsius values to [-1, 1]."""
        return (2.0 * (tensor - self.temp_min) / (self.temp_max - self.temp_min) - 1.0 )

    def _to_celsius(self, tensor):
        """Convert [-1, 1] model outputs back to Celsius."""
        return ((tensor + 1.0) * (self.temp_max - self.temp_min) / 2.0 + self.temp_min)

    def _prepare_batch(self, batch):

        lr = self._to_physical(batch["lr"][:, 0:1], self.lr_mean, self.lr_std)
        hr = self._to_physical(batch["hr"][:, 0:1], self.hr_mean, self.hr_std)

        # Map both images to [-1, 1].
        lr = self._to_model_scale(lr)
        hr = self._to_model_scale(hr)

        # Upsample LR and retain FastDiffSR's original 3-channel representation.
        lr_up = F.interpolate(
            lr,
            size=hr.shape[-2:],
            mode="bicubic",
            align_corners=False,
        ).repeat(1, 3, 1, 1)

        hr = hr.repeat(1, 3, 1, 1)

        # Multiscale auxiliary data for SPADE.
        aux_lr = batch["aux_lr"].float()
        aux_mid = batch["aux_mid"].float()
        aux_hr = batch["aux_hr"].float()

        # Append temporal channels to HR-resolution auxiliary data
        # for the diffusion denoiser input.
        time_channels = batch["time_channels"].float()
        aux_input = torch.cat([aux_hr, time_channels], dim=1)

        # Validate expected channel counts and resolutions.
        assert aux_lr.shape[1] == 23, aux_lr.shape
        assert aux_mid.shape[1] == 23, aux_mid.shape
        assert aux_hr.shape[1] == 23, aux_hr.shape
        assert aux_input.shape[1] == 25, aux_input.shape
        assert aux_lr.shape[-2:] == (64, 64), aux_lr.shape
        assert aux_mid.shape[-2:] == (128, 128), aux_mid.shape
        assert aux_hr.shape[-2:] == hr.shape[-2:], aux_hr.shape
        assert time_channels.shape[-2:] == hr.shape[-2:]

        hr_mask = batch["hr_mask"].float()

        return (
            lr_up,hr,hr_mask, aux_input,time_channels,
            aux_lr, aux_mid,aux_hr,
        )

    # --------- Training---------
    def _training_loss(self, hr,lr_up, hr_mask, water_mask, aux, time_channels,aux_lr, aux_mid, aux_hr,):
        diffusion = self.netG
        # Original FastDiffSR residual and continuous noise-level sampling.
        x_start = diffusion.img2res(hr, lr_up)
        batch_size = x_start.shape[0]
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

        # Denoiser input: LR + noisy residual + auxiliary/temporal data.
        denoiser_input = torch.cat([lr_up, x_noisy, aux], dim=1)
        noise_pred = diffusion.denoise_fn(denoiser_input,sqrt_alpha.view(batch_size, -1),
                                          aux_lr,aux_mid, aux_hr, )

        # Original masked noise-prediction loss.
        diffusion_mask = hr_mask.repeat(1, 3, 1, 1).to(noise_pred.dtype)
        diffusion_error = (noise_pred - noise).abs() * diffusion_mask
        loss_diffusion = (diffusion_error.sum() / diffusion_mask.sum().clamp_min(1.0))

        # Estimate the clean residual using the same continuous alpha.
        noise_scale = torch.sqrt((1.0 - sqrt_alpha.square()).clamp_min(0.0))
        x_start_pred = (x_noisy - noise_scale * noise_pred) / sqrt_alpha.clamp_min(1e-8)

        # Reconstruct SR and apply the shared water/non-water/gradient loss.
        sr_pred = diffusion.res2img(x_start_pred, lr_up)[:, 0:1]
        hr_target = hr[:, 0:1]

        recon = combined_loss(
            sr=sr_pred,
            hr=hr_target,
            hr_mask=hr_mask,
            water_mask=water_mask,
            lambda_nw=self.lambda_nw,
            lambda_w=self.lambda_w,
            lambda_g=self.lambda_g,
            time_channels=time_channels,
        )

        loss_total = loss_diffusion + recon["loss_total"]

        return {
            "total": loss_total,
            "diffusion": loss_diffusion,
            "reconstruction": recon["loss_total"],
            "water": recon["loss_water"],
            "nonwater": recon["loss_nw"],
            "gradient": recon["loss_grad"],
            "temporal_weight": recon["w_time"],
        }

    def training_step(self, batch, batch_idx):
        (lr_up,hr,hr_mask,aux,time_channels,aux_lr,aux_mid,aux_hr,) = self._prepare_batch(batch)
        water_mask = batch["water_mask"].float().to(hr.device)

        losses = self._training_loss(
            hr,
            lr_up,
            hr_mask,
            water_mask,
            aux,
            time_channels,
            aux_lr,
            aux_mid,
            aux_hr,
        )

        self.log("train/loss", losses["total"], prog_bar=True, on_epoch=True)
        self.log("train/diffusion_loss", losses["diffusion"], on_epoch=True)
        self.log("train/reconstruction_loss", losses["reconstruction"], on_epoch=True)
        self.log("train/water_loss", losses["water"], on_epoch=True)
        self.log("train/nonwater_loss", losses["nonwater"], on_epoch=True)
        self.log("train/gradient_loss", losses["gradient"], on_epoch=True)
        self.log("train/temporal_weight", losses["temporal_weight"], on_epoch=True)

        return losses["total"]

    # --------- Auxiliary-conditioned FastDiffSR sampling ---------
    def _p_mean_variance(self,x,t,condition_x,
                         aux,aux_lr,aux_mid,aux_hr,clip_denoised=True, 
                        ):
        diffusion = self.netG
        batch_size = x.shape[0]

        noise_level = torch.full(
            (batch_size, 1),
            float(diffusion.sqrt_alphas_cumprod_prev[t + 1]),
            device=x.device,
            dtype=x.dtype,
        )

        denoiser_input = torch.cat([condition_x, x, aux], dim=1)
        noise_pred = diffusion.denoise_fn(denoiser_input,noise_level,aux_lr,aux_mid,aux_hr,)

        x_recon = diffusion.predict_start_from_noise(x, t=t, noise=noise_pred)

        if clip_denoised:
            x_recon = x_recon.clamp(-1.0, 1.0)

        return diffusion.q_posterior(
            x_start=x_recon,
            x_t=x,
            t=t,
        )

    @torch.no_grad()
    def _p_sample(self, x, t, condition_x,
                  aux,aux_lr,aux_mid, aux_hr,clip_denoised=True, 
                  ):
        model_mean, model_log_variance = self._p_mean_variance(
            x, t,condition_x,
            aux,aux_lr,aux_mid,aux_hr,
            clip_denoised=clip_denoised,
        )

        noise = torch.randn_like(x) if t > 0 else torch.zeros_like(x)
        return model_mean + noise * (0.5 * model_log_variance).exp()


    @torch.no_grad()
    def _super_resolution(self, lr_up, aux, aux_lr, aux_mid, aux_hr):
        diffusion = self.netG
        residual = torch.randn_like(lr_up)

        for t in reversed(range(diffusion.num_timesteps)):
            residual = self._p_sample(
                residual,
                t,
                lr_up,
                aux,
                aux_lr,
                aux_mid,
                aux_hr,
            )

        return diffusion.res2img(residual, lr_up)


    def forward(self, lr_up, aux, aux_lr, aux_mid, aux_hr):
        return self._super_resolution( lr_up, aux,aux_lr,aux_mid,aux_hr,)
    
    # ---------- Validation and testing ------------
    def _evaluation_step(self, batch, stage):
        (lr_up,hr_scaled,hr_mask,aux, _, aux_lr,aux_mid,aux_hr,) = self._prepare_batch(batch)

        sr_scaled = self( lr_up,aux, aux_lr, aux_mid,aux_hr,)

        sr = self._to_celsius(sr_scaled[:, 0:1])
        hr = self._to_celsius(hr_scaled[:, 0:1])

        water_mask = batch.get("water_mask")
        if water_mask is not None:
            water_mask = water_mask.to(sr.device)

        metrics = compute_metrics(
            module=self,
            sr=sr,
            hr=hr,
            hr_mask=hr_mask,
            stage=stage,
            water_mask=water_mask,
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

    # ---------Optimizer and scheduler ----------------

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