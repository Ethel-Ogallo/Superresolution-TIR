"""
film_aux.py — FiLM conditioning module for TIR super-resolution.

AUX content: spectral bands (B2,B3,B4,B5,B8,B11), spectral indices
(NDVI, NDMI, NDWI), and one-hot LULC from ESA WorldCover.

All AUX patches are preprocessed to 7.5m resolution and spatially
aligned to the HR thermal patch during dataset creation — so AUX
and HR share the same grid with perfect pixel correspondence.

Encoding happens directly at HR scale. No downsampling is needed or
applied. The interpolate guard in forward() exists only as a safety
net for rare edge-case size mismatches.

InstanceNorm2d is used instead of BatchNorm2d because:
  - BatchNorm accumulates running statistics across training batches
    and switches to those at test time — if stats are poorly estimated
    (e.g. AUX patches vary heavily across scenes) this causes NaN loss
  - InstanceNorm normalises each sample independently, has no running
    statistics, and behaves identically at train and test time

Backbone-agnostic: call film(aux, target_size=res.shape[-2:]) from
any forward method (EDSR, SwinIR, HAT, RealESRGAN, etc.).

Shape contract:
  aux         : (B, C_aux,         H_hr,   W_hr)
  target_size : (H_feat, W_feat)
  gamma, beta : (B, feat_channels, H_feat, W_feat)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class AuxFiLM(nn.Module):
    def __init__(
        self,
        aux_channels: int,
        feat_channels: int,
        hidden: int = 64,
    ):
        """
        Args:
            aux_channels:  total aux input channels (cont bands + one-hot LULC)
            feat_channels: backbone feature channels to modulate
            hidden:        internal channel width
        """
        super().__init__()
        self.feat_channels = feat_channels

        # Encode spatial structure at HR resolution.
        # InstanceNorm2d: normalises per sample per channel — no running stats,
        # no train/test discrepancy, robust to scene-to-scene AUX variation.
        self.encoder = nn.Sequential(
            nn.Conv2d(aux_channels, hidden, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(hidden, hidden, 3, padding=1, bias=False),
            nn.InstanceNorm2d(hidden, affine=True),
            nn.ReLU(inplace=True),
        )

        # Predict gamma and beta
        self.head = nn.Conv2d(hidden, feat_channels * 2, 3, padding=1)

        # Learned scale on gamma — lets the network suppress features
        # beyond the (0, 2) range that tanh+1 alone would allow
        self.gamma_scale = nn.Parameter(torch.ones(1, feat_channels, 1, 1))

        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, a=0.0, mode="fan_out",
                                        nonlinearity="relu")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.InstanceNorm2d):
                if m.weight is not None:
                    nn.init.ones_(m.weight)
                if m.bias is not None:
                    nn.init.zeros_(m.bias)

        # Zero-init head so FiLM starts as identity:
        # gamma_raw=0 -> tanh(0)=0 -> gamma=1*1=1, beta=tanh(0)=0
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def forward(self, aux: torch.Tensor, target_size: tuple) -> tuple:
        """
        Args:
            aux:         (B, C_aux, H_hr, W_hr) — AUX at 7.5m, aligned to HR
            target_size: (H_feat, W_feat)        — res.shape[-2:] in the backbone

        Returns:
            gamma: (B, feat_channels, H_feat, W_feat)
            beta:  (B, feat_channels, H_feat, W_feat)
        """
        # Encode directly at HR scale — perfect pixel correspondence with res
        x = self.encoder(aux)

        # Safety guard only — in practice aux and target_size match exactly
        # since both are at 7.5m on the same aligned grid
        if (aux.shape[-2], aux.shape[-1]) != target_size:
            x = F.interpolate(
                x, size=target_size, mode="bilinear", align_corners=False
            )

        out = self.head(x)
        gamma_raw, beta = torch.chunk(out, 2, dim=1)

        gamma = self.gamma_scale * (1.0 + torch.tanh(gamma_raw))
        beta  = torch.tanh(beta)

        return gamma, beta