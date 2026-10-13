"""

FastDiffSR x4 Lightning adapter for thermal infrared (TIR) super-resolution.
Original implementation: https://github.com/Meng-333/FastDiffSR
Paper: 10.1109/TGRS.2024.3458009

"""

from pathlib import Path
import json
import sys
import numpy as np
import torch
import torch.nn.functional as F
import lightning.pytorch as pl

# --------------- Paths and constants ------------------
SISR_ROOT = Path(__file__).resolve().parents[2]   # Superresolution-TIR/SISR
PROJECT_ROOT = SISR_ROOT.parent                   # Superresolution-TIR

sys.path.insert(0, str(SISR_ROOT)) # Allow `import scripts.utils.metrics`
 
FASTDIFFSR_ROOT = SISR_ROOT / "scripts" / "external" / "FastDiffSR"
CONFIG_PATH = FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json"
CHECKPOINT_PATH = PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"
 

# ---------------- helper functions ----------------
def build_diffusion_model(config_path):
    """Create the FastDiffSR network (UNet + diffusion wrapper) from the config."""

    sys.path.insert(0, str(FASTDIFFSR_ROOT.parent))
    from FastDiffSR.fastdiffsr_modules import diffusion, unet
 
    with open(config_path, "r") as f:
        config = json.load(f)
 
    unet_opt = config["model"]["unet"]
    diff_opt = config["model"]["diffusion"]
 
    # check model matches the pretrained checkpoint
    assert diff_opt["image_size"] == 256
    assert diff_opt["channels"] == 3
    assert unet_opt["in_channel"] == 6
    assert unet_opt["out_channel"] == 3
 
    # The UNet 
    denoiser = unet.UNet(
        in_channel=unet_opt["in_channel"],
        out_channel=unet_opt["out_channel"],
        norm_groups=unet_opt.get("norm_groups", 32),
        inner_channel=unet_opt["inner_channel"],
        channel_mults=unet_opt["channel_multiplier"],
        attn_res=unet_opt["attn_res"],
        res_blocks=unet_opt["res_blocks"],
        dropout=unet_opt["dropout"],
        image_size=diff_opt["image_size"],
    )
 
    # GaussianDiffusion wraps the UNet
    netG = diffusion.GaussianDiffusion(
        denoiser,
        image_size=diff_opt["image_size"],
        channels=diff_opt["channels"],
        loss_type="l1",
        conditional=diff_opt["conditional"],
        schedule_opt=None,
        scale=4,
    )
 
    # set up noise schedule before loading weights
    netG.set_new_noise_schedule(
        config["model"]["beta_schedule"]["train"],
        device=torch.device("cpu"),
    )
    netG.set_loss(torch.device("cpu"))
 
    return netG
 
 
def load_pretrained_weights(netG, checkpoint_path):
    """Load the pretrained checkpoint and fail loudly if anything does not match."""
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)

    extra_keys = {k for k in checkpoint if k.startswith("feature_extractor.")} # feature extractor not included in code
    weights = {k: v for k, v in checkpoint.items() if k not in extra_keys}
 
    model_state = netG.state_dict()
    missing = set(model_state) - set(weights)
    unexpected = set(weights) - set(model_state)
    shape_errors = [
        (k, tuple(model_state[k].shape), tuple(weights[k].shape))
        for k in model_state.keys() & weights.keys()
        if model_state[k].shape != weights[k].shape
    ]
 
    if missing or unexpected or shape_errors:
        raise RuntimeError(
            "Checkpoint mismatch:\n"
            f"Missing keys: {sorted(missing)}\n"
            f"Unexpected keys: {sorted(unexpected)}\n"
            f"Shape mismatches: {shape_errors}"
        )
 
    netG.load_state_dict(weights, strict=True)
    print(
        f"[FastDiffSR] Checkpoint loaded: {len(weights)} parameters; "
        f"excluded {len(extra_keys)} unused feature-extractor parameters."
    )


# --------------- Model Definition ----------------
class FastDiffSRModule(pl.LightningModule):
    """
    Lightning adapter for pretrained FastDiffSR x4.
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
        self.save_hyperparameters(ignore=["checkpoint_path", "config_path"])
 
        # --- Dataset statistics ---
        with open(stats_path, "r") as f:
            stats = json.load(f)
 
        # Used to undo SRDataset's z-scoring.
        self.hr_mean = float(stats["hr"]["mean"])
        self.hr_std = float(stats["hr"]["std"])
        self.lr_mean = float(stats["lr"]["mean"])
        self.lr_std = float(stats["lr"]["std"])
        self.temp_min = float(min(stats["hr"]["min"], stats["lr"]["min"]))
        self.temp_max = float(max(stats["hr"]["max"], stats["lr"]["max"]))
 
        # Used ONLY by compute_metrics
        self.DATA_MIN = float(stats["hr_percentiles"]["p1"])
        self.DATA_RANGE = float(stats["hr_data_range"])
 
        # --- Network ---
        self.netG = build_diffusion_model(config_path)
        load_pretrained_weights(self.netG, checkpoint_path)
 
    # Temperature scaling helpers
    @staticmethod
    def _single_channel(tensor): # keep one copy of TIR
        return tensor[:, 0:1]
 
    @staticmethod
    def _to_physical(tensor, mean, std):
        """turns z-score to Celsius"""
        return tensor * std + mean
 
    def _to_model_scale(self, tensor):
        """maps celsius to [-1, 1] for both LR and HR"""
        return 2.0 * (tensor - self.temp_min) / (self.temp_max - self.temp_min) - 1.0
 
    def _to_celsius(self, tensor):
        """inverse of _to_model_scale"""
        return (tensor + 1.0) * (self.temp_max - self.temp_min) / 2.0 + self.temp_min
 
    # Batch preparation
    def _prepare_batch(self, batch):
        """Turn a dataset batch into (upsampled LR, HR, HR mask) on the model scale."""

        lr_c = self._to_physical(self._single_channel(batch["lr"]), self.lr_mean, self.lr_std)
        hr_c = self._to_physical(self._single_channel(batch["hr"]), self.hr_mean, self.hr_std)
 
        lr = self._to_model_scale(lr_c)
        hr = self._to_model_scale(hr_c)
 
        # FastDiffSR conditions on a full-size (blurry) LR image.
        lr_up = F.interpolate(lr, size=hr.shape[-2:], mode="bicubic", align_corners=False)
 
        # The pretrained UNet expects 3-channel images
        lr_up = lr_up.repeat(1, 3, 1, 1)
        hr = hr.repeat(1, 3, 1, 1)
 
        return lr_up, hr, batch["hr_mask"].float()
 
    # Masked Loss (L1)
    def _masked_diffusion_loss(self, hr, lr_up, hr_mask):
        """
        Follows p_losses() of the original code, but ignores invalid HR pixels.
        """
        diffusion = self.netG
 
        # residual before addition of noise
        x_start = diffusion.img2res(hr, lr_up)
        batch_size = x_start.shape[0]
 
        # Random noise level: pick a step t, then a continuous strength between steps t-1 and t for each image in the batch.
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
 
        # Corrupt the clean residual with known noise.
        noise = torch.randn_like(x_start)
        x_noisy = diffusion.q_sample(
            x_start=x_start,
            continuous_sqrt_alpha_cumprod=sqrt_alpha,
            noise=noise,
        )
 
        noise_pred = diffusion.denoise_fn(
            torch.cat([lr_up, x_noisy], dim=1),
            sqrt_alpha.view(batch_size, -1),
        )
 
        # L1 error between guessed and true noise, valid pixels only.
        mask = hr_mask.repeat(1, 3, 1, 1).to(noise_pred.dtype)
        abs_error = (noise_pred - noise).abs() * mask
        return abs_error.sum() / mask.sum().clamp_min(1.0)
 
    # forward, training, validation, and test steps
    def forward(self, lr_up):
        """Inference: denoise from pure noise, conditioned on LR (one image at a time)."""
        if lr_up.shape[0] == 1:
            return self.netG.super_resolution(lr_up)
 
        outputs = [self.netG.super_resolution(lr_up[i:i + 1]) for i in range(lr_up.shape[0])]
        return torch.cat(outputs, dim=0)
 
    def training_step(self, batch, batch_idx):
        lr_up, hr, hr_mask = self._prepare_batch(batch)
        loss = self._masked_diffusion_loss(hr, lr_up, hr_mask)
 
        self.log(
            "train/loss", loss,
            on_step=True, on_epoch=True, prog_bar=True,
            batch_size=hr.shape[0],
        )
        return loss
 
    def _evaluation_step(self, batch, stage):
        lr_up, hr_scaled, hr_mask = self._prepare_batch(batch)
 
        with torch.no_grad():
            sr_scaled = self(lr_up)
 
        # compute_metrics expects physical units (Celsius).
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
            if value is None:        # tile had too few valid pixels
                continue
            if not isinstance(value, torch.Tensor):
                value = torch.tensor(value, device=self.device, dtype=torch.float32)
 
            self.log(
                f"{stage}/{name}", value,
                on_step=False, on_epoch=True,
                prog_bar=(name == "water_mae"),
                batch_size=hr.shape[0],
            )
 
    def validation_step(self, batch, batch_idx):
        self._evaluation_step(batch, "val")
 
    def test_step(self, batch, batch_idx):
        self._evaluation_step(batch, "test")
 
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.netG.parameters(),
            lr=self.hparams.learning_rate,
            weight_decay=self.hparams.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
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
