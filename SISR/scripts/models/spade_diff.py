"""
spade_diff.py — Spatially Adaptive Denormalization for TIR Super-Resolution
adapted from: https://github.com/NVlabs/SPADE
"""

import torch
import torch.nn as nn


class SPADE(nn.Module):
    """
    Spatially Adaptive Feature Modulation.
    """
    def __init__(self, norm_nc: int, label_nc: int = 14, ks: int = 3):
        super().__init__()

        padding = ks // 2

        # Project auxiliary information into a shared feature space
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=ks, padding=padding,),
            nn.ReLU(inplace=True),)

        # Predict spatial gamma and beta
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=ks, padding=padding,)
        self.mlp_beta = nn.Conv2d(128, norm_nc, kernel_size=ks,padding=padding,)

        # Initialise modulation close to identity
        nn.init.zeros_(self.mlp_gamma.weight)
        nn.init.zeros_(self.mlp_gamma.bias)

        nn.init.zeros_(self.mlp_beta.weight)
        nn.init.zeros_(self.mlp_beta.bias)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:

        assert x.ndim == 4, f"Expected 4D feature map, got {x.shape}"
        assert seg.ndim == 4, f"Expected 4D auxiliary map, got {seg.shape}"

        assert x.shape[0] == seg.shape[0], (
            f"Batch mismatch: features {x.shape}, auxiliary {seg.shape}"
        )

        assert x.shape[-2:] == seg.shape[-2:], (
            f"Spatial mismatch: features {x.shape}, auxiliary {seg.shape}"
        )

        # Generate spatial conditioning features
        actv = self.mlp_shared(seg)

        gamma = self.mlp_gamma(actv)
        beta = self.mlp_beta(actv)

        # Apply spatial modulation to already-normalised features
        return x * (1.0 + gamma) + beta