# spade.py — Spatially Adaptive Denormalization for TIR Super-Resolution
import torch
import torch.nn as nn
from torch.nn.utils import spectral_norm
import torch.nn.functional as F

# -------------------- SPADE LAYER ------------------------
class SPADE(nn.Module):
    """
    Spatially Adaptive Denormalization layer.
    Maps an heterogeneous continuous-categorical geospatial auxiliary stack
    to adaptive pixel-specific scale and shift vectors.
    """
    def __init__(self, norm_nc: int, label_nc: int, ks: int = 3):
        super().__init__()

        # Instance normalization WITHOUT affine parameters
        self.param_free_norm = nn.InstanceNorm2d(norm_nc, affine=False)

        # Shared convolution to project continuous/categorical geodata 
        pw = ks // 2
        self.mlp_shared = nn.Sequential(
            nn.Conv2d(label_nc, 128, kernel_size=ks, padding=pw),
            nn.ReLU(inplace=True),
        )

        # Predict spatial modulation parameters with deep intermediate activations
        # Park et al. intent: Convolutions mapping embeddings to gamma/beta MUST have 
        # non-linear transformations contextually guiding them.
        self.mlp_gamma = nn.Sequential(
            nn.Conv2d(128, norm_nc, kernel_size=ks, padding=pw)
        )
        self.mlp_beta = nn.Sequential(
            nn.Conv2d(128, norm_nc, kernel_size=ks, padding=pw)
        )

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # Enforce hard dataset/model structural alignment
        assert seg.shape[-2:] == x.shape[-2:], (
            f"SPADE spatial mismatch: seg {seg.shape} vs x {x.shape}"
        )

        # 1. Parameter-free normalization across the spatial grid
        x_norm = self.param_free_norm(x)

        # 2. Generate non-linear spatial embeddings
        actv = self.mlp_shared(seg)
        gamma = self.mlp_gamma(actv)
        beta  = self.mlp_beta(actv)

        # 3. Element-wise Spatially Adaptive Modulation
        return x_norm * (1 + gamma) + beta


# ---------------- SPADE RESIDUAL BLOCK -----------------
class SPADEResnetBlock(nn.Module):
    """
    Authentic Pre-Activation SPADE Residual Block.
    Structure:
        x → LeakyReLU → SPADE_0 → Conv_0 → LeakyReLU → SPADE_1 → Conv_1 → + skip
    """
    def __init__(self, fin: int, fout: int, seg_nc: int):
        super().__init__()

        self.learned_shortcut = (fin != fout)
        fmiddle = min(fin, fout)

        # Convolutions are spectrally normalized to sustain adversarial stability
        self.conv_0 = spectral_norm(nn.Conv2d(fin, fmiddle, kernel_size=3, padding=1))
        self.conv_1 = spectral_norm(nn.Conv2d(fmiddle, fout, kernel_size=3, padding=1))

        # Reconstructed Pre-Activation Normalization Sequence
        self.norm_0 = SPADE(fin, seg_nc)
        self.norm_1 = SPADE(fmiddle, seg_nc)

        if self.learned_shortcut:
            self.conv_s = spectral_norm(nn.Conv2d(fin, fout, kernel_size=1, bias=False))
            self.norm_s = SPADE(fin, seg_nc)

    def forward(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        # 1. Main Path: Executing authentic Pre-Activation flow
        # Step A: Activation -> SPADE Norm -> Conv
        dx = F.leaky_relu(x, 0.2)
        dx = self.conv_0(self.norm_0(dx, seg))
        
        # Step B: Activation -> SPADE Norm -> Conv
        dx = F.leaky_relu(dx, 0.2)
        dx = self.conv_1(self.norm_1(dx, seg))

        # 2. Skip Connection Path
        x_s = self._shortcut(x, seg)

        return x_s + dx

    def _shortcut(self, x: torch.Tensor, seg: torch.Tensor) -> torch.Tensor:
        if self.learned_shortcut:
            # Shortcut path must also match the pre-activation state space
            x_s = F.leaky_relu(x, 0.2)
            return self.conv_s(self.norm_s(x_s, seg))
        return x


# --------------------- RRDB + SPADE INTEGRATION -----------------------
class RRDBNetWithSPADE(nn.Module):
    """
    Wrapper around RRDBNet that injects SPADE at upsampling stages.
    Maintains a 100% plug-and-play surface with Lightning module.
    """
    def __init__(self, rrdb_net, n_feats=64, seg_nc_mid=20, seg_nc_hr=20):
        super().__init__()
        self.net = rrdb_net
        self.spade_mid = SPADEResnetBlock(n_feats, n_feats, seg_nc_mid)
        self.spade_hr  = SPADEResnetBlock(n_feats, n_feats, seg_nc_hr)

    def forward(self, x, aux_mid=None, aux_hr=None):
        if aux_mid is None or aux_hr is None:
            return self.net(x)

        n = self.net
        # Feature extraction (LR space)
        feat = n.lrelu(n.conv_first(x))
        body_feat = n.conv_body(n.body(feat))
        feat = feat + body_feat

        # Upsample 1 → 128x128 + SPADE
        feat = n.lrelu(n.conv_up1(
            F.interpolate(feat, scale_factor=2, mode="nearest")
        ))
        feat = self.spade_mid(feat, aux_mid)

        # Upsample 2 → 256x256 + SPADE
        feat = n.lrelu(n.conv_up2(
            F.interpolate(feat, scale_factor=2, mode="nearest")
        ))
        feat = self.spade_hr(feat, aux_hr)

        # Output head
        feat = n.lrelu(n.conv_hr(feat))
        return n.conv_last(feat)