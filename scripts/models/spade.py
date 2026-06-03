"""
spade.py — Spatially Adaptive Denormalization for TIR Super-Resolution

This is a corrected and efficiency-aware implementation of SPADE.

Key idea:
    SPADE modulates normalized feature maps using spatially aligned
    conditioning maps (auxiliary data), producing per-pixel γ (scale)
    and β (shift).

Main correction in this version:
    ❌ Removed runtime interpolation inside SPADE (costly + redundant)
    ✔ Assumes aux maps are pre-aligned in dataset/model pipeline
"""

import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm


# =============================================================================
# SPADE LAYER
# =============================================================================
class SPADE(nn.Module):
    """
    Spatially Adaptive Denormalization layer.

    Formula:
        x_norm = InstanceNorm(x)
        output = x_norm * (1 + gamma(seg)) + beta(seg)

    where gamma and beta are learned from conditioning map.
    """

    def __init__(self, norm_nc: int, label_nc: int, ks: int = 3):
        super().__init__()

        # Instance normalization WITHOUT affine parameters
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # Shared convolution to process conditioning map
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True),
        )

        # Predict spatial modulation parameters
        self.mlp_gamma = nn.Conv2d(128, norm_nc, kernel_size=ks, padding=pw)
        self.mlp_beta  = nn.Conv2d(128, norm_nc, kernel_size=ks, padding=pw)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x   : feature map [B, C, H, W]
            seg : conditioning map [B, C_aux, H, W] (MUST match spatial size)

        Returns:
            modulated feature map [B, C, H, W]
        """

        # ---------------------------------------------------------------------
        # SAFETY CHECK (IMPORTANT FIX)
        # ---------------------------------------------------------------------
        # Instead of expensive interpolation every forward pass,
        # we enforce correct alignment at dataset/model level.
        assert seg.shape[-2:] == x.shape[-2:], (
            f"SPADE spatial mismatch: seg {seg.shape} vs x {x.shape}"
        )

        # ---------------------------------------------------------------------
        # Normalization step
        # ---------------------------------------------------------------------
        x_norm = self.param_free_norm(x)

        # ---------------------------------------------------------------------
        # Generate spatial modulation parameters
        # ---------------------------------------------------------------------
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)

        # ---------------------------------------------------------------------
        # SPADE modulation
        # ---------------------------------------------------------------------
        return x_norm * (1 + gamma) + beta


# =============================================================================
# SPADE RESIDUAL BLOCK
# =============================================================================
class SPADEResnetBlock(nn.Module):
    """
    Residual block using SPADE instead of BatchNorm.

    Structure:
        x → SPADE → Conv → ReLU → SPADE → Conv → + skip
    """

    def __init__(self, fin: int, fout: int, seg_nc: int):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # Spectrally normalized convolutions (GAN stability)
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, 3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, 3, padding=1))

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, 1, bias=False))

        # SPADE normalization layers
        self.norm_0 = SPADE(fin, seg_nc)
        self.norm_1 = SPADE(fmiddle, seg_nc)

        if self.learned_shortcut:
            self.norm_s = SPADE(fin, seg_nc)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:

        # Skip connection
        x_s = self._shortcut(x, seg)

        # Main path
        dx = self.conv_0(torch.nn.functional.leaky_relu(self.norm_0(x, seg), 0.2))
        dx = self.conv_1(torch.nn.functional.leaky_relu(self.norm_1(dx, seg), 0.2))

        return x_s + dx

    def _shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            return self.conv_s(self.norm_s(x, seg))
        return x


# =============================================================================
# RRDB + SPADE INTEGRATION
# =============================================================================
class RRDBNetWithSPADE(nn.Module):
    """
    Wrapper around RRDBNet that injects SPADE at upsampling stages.

    Key idea:
        - RRDB learns deep features (texture / structure)
        - SPADE injects spatial conditioning during resolution increase
    """

    def __init__(self, rrdb_net, n_feats=64, seg_nc_mid=20, seg_nc_hr=20):
        super().__init__()

        self.net = rrdb_net

        self.spade_mid = SPADEResnetBlock(n_feats, n_feats, seg_nc_mid)
        self.spade_hr  = SPADEResnetBlock(n_feats, n_feats, seg_nc_hr)

    def forward(self, x, aux_mid, aux_hr):

        n = self.net

        # ---------------------------------------------------------------------
        # Feature extraction (LR space)
        # ---------------------------------------------------------------------
        feat = n.lrelu(n.conv_first(x))
        body_feat = n.conv_body(n.body(feat))
        feat = feat + body_feat

        # ---------------------------------------------------------------------
        # Upsample 1 → 128x128 + SPADE
        # ---------------------------------------------------------------------
        feat = n.lrelu(n.conv_up1(
            torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
        ))
        feat = self.spade_mid(feat, aux_mid)

        # ---------------------------------------------------------------------
        # Upsample 2 → 256x256 + SPADE
        # ---------------------------------------------------------------------
        feat = n.lrelu(n.conv_up2(
            torch.nn.functional.interpolate(feat, scale_factor=2, mode="nearest")
        ))
        feat = self.spade_hr(feat, aux_hr)

        # ---------------------------------------------------------------------
        # Output head
        # ---------------------------------------------------------------------
        feat = n.lrelu(n.conv_hr(feat))
        return n.conv_last(feat)