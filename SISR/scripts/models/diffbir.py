"""

DiffBIR x4 Lightning adapter for thermal infrared (TIR) super-resolution.
Original implementation: https://github.com/XPixelGroup/DiffBIR
Paper: http://arxiv.org/abs/2308.15070

"""

from __future__ import annotations

import json
from pathlib import Path
import sys

import lightning.pytorch as pl
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from omegaconf import OmegaConf


# Project paths
SISR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SISR_ROOT.parent

DIFFBIR_ROOT = SISR_ROOT / "scripts" / "external" / "DiffBIR"

if str(DIFFBIR_ROOT) not in sys.path:
    sys.path.insert(0, str(DIFFBIR_ROOT))

# Import the official DiffBIR implementations.
from diffbir.utils.common import instantiate_from_config
from diffbir.sampler.spaced_sampler import SpacedSampler


# Lightning module
class DiffBIRModule(pl.LightningModule):
    """
    DiffBIR-v2 adapted for paired thermal infrared super-resolution.

    Normalization:
        The dataset's z-score normalization is reversed before converting
        temperatures to the [0, 1] range expected by BSRNet and to the
        [-1, 1] range expected by the Stable Diffusion VAE.
    """

    def __init__(
        self,
        stats_json: str,
        learning_rate: float = 1e-5,
        weight_decay: float = 1e-4,
        sample_steps: int = 20,
        sd_checkpoint: str = "data/pretrained/v2-1_512-ema-pruned.ckpt",
        controlnet_checkpoint: str = "data/pretrained/v2.pth",
        bsrnet_checkpoint: str = "data/pretrained/BSRNet.pth",
        cldm_config: str = ("SISR/scripts/external/DiffBIR/configs/inference/cldm.yaml"),
        diffusion_config: str = ("SISR/scripts/external/DiffBIR/configs/inference/diffusion.yaml"),
        bsrnet_config: str = ("SISR/scripts/external/DiffBIR/configs/inference/bsrnet.yaml"),
    ):
        super().__init__()

        self.save_hyperparameters()
        self.learning_rate = learning_rate
        self.weight_decay = weight_decay
        self.sample_steps = sample_steps

        # Load temperature normalization statistics.
        stats_path = Path(stats_json)
        if not stats_path.is_absolute():
            stats_path = PROJECT_ROOT / stats_path

        with open(stats_path, "r", encoding="utf-8") as f:
            stats = json.load(f)

        self.hr_mean = float(stats["hr"]["mean"])
        self.hr_std = float(stats["hr"]["std"])
        self.lr_mean = float(stats["lr"]["mean"])
        self.lr_std = float(stats["lr"]["std"])

        self.temp_min = float(min(stats["hr"]["min"], stats["lr"]["min"]))
        self.temp_max = float(max(stats["hr"]["max"], stats["lr"]["max"]))

        if self.temp_max <= self.temp_min:
            raise ValueError("Invalid temperature range in stats JSON.")

        # Resolve checkpoint/config paths.
        def resolve_project_path(path: str) -> Path:
            p = Path(path)
            return p if p.is_absolute() else PROJECT_ROOT / p

        sd_path = resolve_project_path(sd_checkpoint)
        controlnet_path = resolve_project_path(controlnet_checkpoint)
        bsrnet_path = resolve_project_path(bsrnet_checkpoint)

        cldm_cfg_path = resolve_project_path(cldm_config)
        diffusion_cfg_path = resolve_project_path(diffusion_config)
        bsrnet_cfg_path = resolve_project_path(bsrnet_config)

        for path in (
            sd_path,
            controlnet_path,
            bsrnet_path,
            cldm_cfg_path,
            diffusion_cfg_path,
            bsrnet_cfg_path,
        ):
            if not path.exists():
                raise FileNotFoundError(f"Required DiffBIR file not found: {path}")

        # Load the BSRNet conditioning network.-
        bsrnet_cfg = OmegaConf.load(bsrnet_cfg_path)
        self.bsrnet = instantiate_from_config(bsrnet_cfg)
        bsrnet_state = torch.load(
            bsrnet_path,
            map_location="cpu",
            weights_only=False,
        )

        # The checkpoint was verified to be a raw state dictionary.
        self.bsrnet.load_state_dict(bsrnet_state, strict=True)
        self.bsrnet.eval()
        for parameter in self.bsrnet.parameters():
            parameter.requires_grad = False

        #  Load the  DiffBIR ControlLDM architecture.
        cldm_cfg = OmegaConf.load(cldm_cfg_path)
        self.cldm = instantiate_from_config(cldm_cfg)

        # Initialize the Stable Diffusion 2.1 backbone.
        sd_checkpoint_data = torch.load(
            sd_path,
            map_location="cpu",
            weights_only=False,
        )

        sd_state = sd_checkpoint_data.get( "state_dict",sd_checkpoint_data,)
        self.cldm.load_pretrained_sd(sd_state)

        # Initialize ControlNet from the DiffBIR-v2 checkpoint.
        controlnet_state = torch.load(
            controlnet_path,
            map_location="cpu",
            weights_only=False,
        )

        controlnet_state = controlnet_state.get(
            "state_dict",
            controlnet_state,
        )

        self.cldm.load_controlnet_from_ckpt(controlnet_state)

        # Load the diffusion process.
        diffusion_cfg = OmegaConf.load(diffusion_cfg_path)
        self.diffusion = instantiate_from_config(diffusion_cfg)

        # This sampler is used for validation/test inference.
        self.sampler = SpacedSampler(
            self.diffusion.betas,
            self.diffusion.parameterization,
            rescale_cfg=False,
        )

        #  Freeze the pretrained components.
        for parameter in self.cldm.parameters():
            parameter.requires_grad = False
        for parameter in self.cldm.controlnet.parameters():
            parameter.requires_grad = True

        self.cldm.eval()
        self.diffusion.eval()

        # Ensure the ControlNet remains trainable.
        self.cldm.controlnet.train()

        print("[DiffBIR] Model initialized.")
        print(f"[DiffBIR] Stable Diffusion: {sd_path}")
        print(f"[DiffBIR] ControlNet: {controlnet_path}")
        print(f"[DiffBIR] BSRNet: {bsrnet_path}")
        print("[DiffBIR] Trainable component: ControlNet")

    #  Normalization helpers
    @staticmethod
    def _single_channel(x: torch.Tensor) -> torch.Tensor:
        """Use the first channel when the dataset repeats TIR three times."""
        if x.ndim != 4:
            raise ValueError(f"Expected [B,C,H,W], received {tuple(x.shape)}")
        return x[:, 0:1]

    def _to_physical(
        self,
        x: torch.Tensor,
        mean: float,
        std: float,
    ) -> torch.Tensor:
        """Reverse dataset z-score normalization to recover Celsius."""
        return x * std + mean

    def _to_unit_interval(self, temperature: torch.Tensor) -> torch.Tensor:
        """Convert Celsius to [0, 1] for BSRNet and Stable Diffusion conditioning."""
        temperature = (temperature - self.temp_min) / (self.temp_max - self.temp_min)
        return temperature.clamp(0.0, 1.0)

    def _to_celsius(self, image: torch.Tensor) -> torch.Tensor:
        """Convert a [0, 1] model output back to Celsius."""
        image = image.clamp(0.0, 1.0)
        return image * (self.temp_max - self.temp_min) + self.temp_min

    def _prepare_batch(self, batch):
        """Convert the dataset's normalized LR/HR tensors into model inputs."""
        lr_z = self._single_channel(batch["lr"])
        hr_z = self._single_channel(batch["hr"])

        lr_celsius = self._to_physical( lr_z,self.lr_mean,self.lr_std,)
        hr_celsius = self._to_physical( hr_z,self.hr_mean,self.hr_std, )

        lr01 = self._to_unit_interval(lr_celsius)
        hr01 = self._to_unit_interval(hr_celsius)

        # BSRNet expects a three-channel image.
        lr01 = lr01.repeat(1, 3, 1, 1)
        hr_mask = batch["hr_mask"].float()

        # Enforce the expected mask convention: valid = 1, invalid = 0.
        hr_mask = (hr_mask > 0.5).float()

        return lr01, hr01, hr_mask, hr_celsius

    # Conditioning
    @torch.no_grad()
    def _make_condition(self, lr01: torch.Tensor):
        clean = self.bsrnet(lr01)
        clean = clean.clamp(0.0, 1.0)  # Ensure the BSRNet output is in [0, 1].

        prompts = [""] * clean.shape[0]

        condition = self.cldm.prepare_condition(
            clean,
            prompts,
        )

        return condition

    #  Masked diffusion loss
    def _masked_diffusion_loss(
        self,
        z_hr: torch.Tensor,
        timesteps: torch.Tensor,
        condition: dict,
        hr_mask: torch.Tensor,
    ) -> torch.Tensor:
        
        noise = torch.randn_like(z_hr)

        z_noisy = self.diffusion.q_sample(
            x_start=z_hr,
            t=timesteps,
            noise=noise,
        )

        prediction = self.cldm(
            z_noisy,
            timesteps,
            condition,
        )

        if self.diffusion.parameterization == "x0":
            target = z_hr
        elif self.diffusion.parameterization == "eps":
            target = noise
        elif self.diffusion.parameterization == "v":
            target = self.diffusion.get_v(
                z_hr,
                noise,
                timesteps,
            )
        else:
            raise NotImplementedError(
                "Unsupported diffusion parameterization: "
                f"{self.diffusion.parameterization}"
            )

        elementwise_loss = self.diffusion.get_loss(
            prediction,
            target,
            mean=False,
        )

        # Map the HR validity mask to the VAE latent resolution.
        latent_mask = F.interpolate(
            hr_mask,
            size=z_hr.shape[-2:],
            mode="nearest",
        )

        latent_mask = latent_mask.expand_as(elementwise_loss)

        numerator = (elementwise_loss * latent_mask).sum()
        denominator = latent_mask.sum().clamp_min(1.0)

        return numerator / denominator

    #  Training
    def training_step(self, batch, batch_idx):
        lr01, hr01, hr_mask, _ = self._prepare_batch(batch)

        # The HR image must be mapped to [-1, 1] before VAE encoding.
        hr_for_vae = hr01 * 2.0 - 1.0

        # Only ControlNet is optimized.
        with torch.no_grad():
            z_hr = self.cldm.vae_encode(
                hr_for_vae,
                sample=False,
            )
            condition = self._make_condition(lr01)

        timesteps = torch.randint(
            low=0,
            high=self.diffusion.num_timesteps,
            size=(z_hr.shape[0],),
            device=z_hr.device,
        )

        loss = self._masked_diffusion_loss(
            z_hr=z_hr,
            timesteps=timesteps,
            condition=condition,
            hr_mask=hr_mask,
        )

        self.log(
            "train/loss",
            loss,
            on_step=True,
            on_epoch=True,
            prog_bar=True,
            batch_size=lr01.shape[0],
        )

        return loss

    #  Foward pass for inference
    @torch.no_grad()
    def forward(self, lr01: torch.Tensor) -> torch.Tensor:
        """
        Produce a ×4 TIR estimate using the official DiffBIR sampler.

        Args:
            lr01: LR image in [0, 1], [B, 3, H, W].

        Returns:
            SR estimate in [0, 1], [B, 1, 4H, 4W].
        """
        condition = self._make_condition(lr01)

        # Unconditional text embedding uses an empty prompt.
        # The image condition is zeroed for classifier-free guidance.
        uncond = {
            "c_txt": condition["c_txt"],
            "c_img": torch.zeros_like(condition["c_img"]),
        }

        batch_size = lr01.shape[0]
        output_height = lr01.shape[-2] * 4
        output_width = lr01.shape[-1] * 4

        # Stable Diffusion's VAE uses an 8x spatial downsampling factor.
        latent_size = (
            batch_size,
            4,
            output_height // 8,
            output_width // 8,
        )

        latent = self.sampler.sample(
            model=self.cldm,
            device=str(lr01.device),
            steps=self.sample_steps,
            x_size=latent_size,
            cond=condition,
            uncond=uncond,
            cfg_scale=1.0,
            progress=False,
        )

        sr = self.cldm.vae_decode(latent)

        # Convert VAE output from [-1, 1] to [0, 1].
        sr = ((sr + 1.0) / 2.0).clamp(0.0, 1.0)

        # TIR is represented by three repeated channels.
        return sr[:, 0:1]

    #  Validation / testing
    def _evaluate(self, batch, stage: str):
        from scripts.utils.metrics import compute_metrics

        lr01, _, hr_mask, hr_celsius = self._prepare_batch(batch)

        # Run diffusion inference at ×4.
        sr01 = self.forward(lr01)

        sr_celsius = self._to_celsius(sr01)
        hr_celsius = hr_celsius[:, 0:1]

        compute_metrics(
            module=self,
            sr=sr_celsius,
            hr=hr_celsius,
            hr_mask=hr_mask,
            stage=stage,
            water_mask=batch.get("water_mask"),
        )

    def validation_step(self, batch, batch_idx):
        self._evaluate(batch, stage="val")

    def test_step(self, batch, batch_idx):
        self._evaluate(batch, stage="test")

    # Optimizer and scheduler
    def configure_optimizers(self):
        optimizer = AdamW(
            self.cldm.controlnet.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )

        scheduler = ReduceLROnPlateau(
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


