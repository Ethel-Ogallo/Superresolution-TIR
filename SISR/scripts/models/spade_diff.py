"""
spade_diff.py — Multiscale SPADE conditioning for FastDiffSR TIR SR.
Adds semantic feature modulation to the original FastDiffSR U-Net.
"""

import torch
import torch.nn as nn


class SPADE(nn.Module):
    """
    Spatially adaptive feature modulation.
    """

    def __init__(self, norm_nc: int, label_nc: int = 14, ks: int = 3):
        super().__init__()

        padding = ks // 2

        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=ks, padding=padding),
            nn.ReLU(inplace=True),
        )

        self.mlp_gamma = nn.Conv2d(
            128, norm_nc, kernel_size=ks, padding=padding
        )

        self.mlp_beta = nn.Conv2d(
            128, norm_nc, kernel_size=ks, padding=padding
        )

        # Initialize modulation close to identity.
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)

        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 4, f"Expected 4D features, got {x.shape}"
        assert seg.ndim == 4, f"Expected 4D semantic map, got {seg.shape}"
        assert x.shape[0] == seg.shape[0], "Batch mismatch"
        assert x.shape[-2:] == seg.shape[-2:], "Spatial mismatch"
        assert seg.shape[1] == self.mlp_shared[0].in_channels, (
            f"Expected {self.mlp_shared[0].in_channels} semantic channels, "
            f"got {seg.shape[1]}"
        )

        seg = seg.to(device=x.device, dtype=x.dtype)

        actv = self.mlp_shared(seg)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        return x * (1.0 + gamma) + beta


class SPADEWrapper(nn.Module):
    """
    Inserts multiscale SPADE modulation into the original FastDiffSR U-Net.

    SPADE locations:
        64 x 64   -> 128 feature channels -> aux_lr
        128 x 128 -> 128 feature channels -> aux_mid
        256 x 256 -> 64 feature channels  -> aux_hr
    """

    def __init__(
        self,
        unet: nn.Module,
        aux_channels: int = 23,
        label_nc: int = 14,
        label_start: int = 8,
    ):
        super().__init__()

        self.unet = unet

        self.aux_channels = aux_channels
        self.label_nc = label_nc
        self.label_start = label_start

        # Identify the original decoder upsampling modules.
        self.upsample_modules = [
            layer
            for layer in self.unet.ups
            if layer.__class__.__name__ == "Upsample"
        ]

        if len(self.upsample_modules) != 3:
            raise ValueError(
                f"Expected 3 decoder upsampling modules, "
                f"found {len(self.upsample_modules)}"
            )

        feature_channels = [
            layer.conv.out_channels
            for layer in self.upsample_modules
        ]

        expected_channels = [256, 256, 128]

        if feature_channels != expected_channels:
            raise ValueError(
                f"Unexpected decoder channels: {feature_channels}. "
                f"Expected {expected_channels}."
            )

        # One trainable SPADE module per decoder resolution.
        self.spade_layers = nn.ModuleList([
            SPADE(norm_nc=channels, label_nc=label_nc)
            for channels in feature_channels
        ])

    def _get_cosia(self, aux: torch.Tensor) -> torch.Tensor:
        """Extract the 14 COSIA semantic channels."""

        # Remove an unexpected singleton dimension, if present.
        if aux.ndim == 5 and aux.shape[2] == 1:
            aux = aux.squeeze(2)

        if aux.ndim != 4:
            raise ValueError(
                f"Expected auxiliary tensor [B,C,H,W], got {aux.shape}"
            )

        if aux.shape[1] != self.aux_channels:
            raise ValueError(
                f"Expected {self.aux_channels} auxiliary channels, "
                f"got {aux.shape[1]}"
            )

        start = self.label_start
        end = start + self.label_nc

        return aux[:, start:end, :, :]

    def forward(
        self,
        x: torch.Tensor,
        time: torch.Tensor,
        aux_lr: torch.Tensor,
        aux_mid: torch.Tensor,
        aux_hr: torch.Tensor,
    ) -> torch.Tensor:

        # Prepare semantic conditioning at the three resolutions.
        semantic_maps = [
            self._get_cosia(aux_lr),
            self._get_cosia(aux_mid),
            self._get_cosia(aux_hr),
        ]

        expected_sizes = [
            (64, 64),
            (128, 128),
            (256, 256),
        ]

        for seg, size in zip(semantic_maps, expected_sizes):
            if seg.shape[-2:] != size:
                raise ValueError(
                    f"Expected semantic map size {size}, "
                    f"got {seg.shape[-2:]}"
                )

        # Original FastDiffSR timestep embedding.
        if self.unet.noise_level_mlp is not None:
            t = self.unet.noise_level_mlp(time)
        else:
            t = None

        #  Original encoder
        feats = []

        for layer in self.unet.downs:

            if layer.__class__.__name__ == "ResnetBlocWithAttn":
                x = layer(x, t)
            else:
                x = layer(x)

            feats.append(x)

        #  Original middle blocks
        for layer in self.unet.mid:
            x = layer(x, t)

        # Original decoder + SPADE
        spade_idx = 0

        for layer in self.unet.ups:

            if layer.__class__.__name__ == "ResnetBlocWithAttn":

                # Preserve the original U-Net skip connections.
                x = torch.cat([x, feats.pop()], dim=1)
                x = layer(x, t)

            elif layer.__class__.__name__ == "Upsample":

                # Original upsampling operation.
                x = layer(x)
                # Apply SPADE at the corresponding resolution.
                x = self.spade_layers[spade_idx](
                    x,
                    semantic_maps[spade_idx],
                )

                spade_idx += 1

            else:
                raise TypeError(
                    f"Unexpected decoder layer: {layer.__class__.__name__}"
                )

        if spade_idx != 3:
            raise RuntimeError(
                f"Expected 3 SPADE applications, got {spade_idx}"
            )

        #  Original output convolution
        return self.unet.final_conv(x)