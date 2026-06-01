"""
SwinIR — Direct input fusion + two-stage SPADE

Architecture (verified against pretrained checkpoint):
  conv_first          [180, 3,   3, 3] → expanded to [180, 21, 3, 3]
  forward_features    6x RSTB blocks   → 180ch throughout
  conv_after_body     [180, 180, 3, 3] → residual add with x_first
  conv_before_upsample[64,  180, 3, 3] + LeakyReLU → 64ch
  upsample_s1         Conv[256,64]+PixelShuffle(2)  → 64ch at 128px
  spade_mid           feat_ch=64, aux_ch=20
  upsample_s2         Conv[256,64]+PixelShuffle(2)  → 64ch at 256px
  spade_hr            feat_ch=64, aux_ch=20
  conv_last           [3, 64, 3, 3]                 → 3ch
  self.out            [1, 3,  1, 1]                 → 1ch TIR

Input fusion: LR TIR (1ch) + aux_lr (20ch) → 21ch into conv_first
Pretrained-mean init: average RGB filters → repeat for all 21ch
"""

import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


# =========================================================
# SPADE BLOCK
# Faithful to NVlabs:
#   - InstanceNorm first (works at any batch size incl. 1)
#   - (1 + gamma) residual → near-identity at init,
#     safe insertion into pretrained backbone
# =========================================================
class SPADE(nn.Module):
    def __init__(self, feat_ch, aux_ch, hidden=128):
        super().__init__()
        self.norm   = nn.InstanceNorm2d(feat_ch, affine=False)
        self.shared = nn.Sequential(
            nn.Conv2d(aux_ch, hidden, 3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, feat_ch, 3, padding=1)
        self.beta  = nn.Conv2d(hidden, feat_ch, 3, padding=1)

    def forward(self, x, aux):
        assert x.shape[-2:] == aux.shape[-2:], (
            f"SPADE spatial mismatch: feat {x.shape[-2:]} vs aux {aux.shape[-2:]}"
        )
        x_norm = self.norm(x)
        h      = self.shared(aux)
        return x_norm * (1 + self.gamma(h)) + self.beta(h)


# =========================================================
# SPLIT UPSAMPLE
# Pretrained upsample = nn.Sequential of 4 layers:
#   [Conv(64→256), PixelShuffle(2), Conv(64→256), PixelShuffle(2)]
# Split into two x2 stages so SPADE fires between them.
# Called BEFORE pretrained load to preserve state dict keys.
# =========================================================
def _split_upsample(upsample_seq):
    layers = list(upsample_seq.children())
    assert len(layers) == 4, (
        f"Expected 4 layers for x4 pixelshuffle, got {len(layers)}"
    )
    return (nn.Sequential(layers[0], layers[1]),   # 64px → 128px
            nn.Sequential(layers[2], layers[3]))   # 128px → 256px


# =========================================================
# SWINIR MODULE
# =========================================================
class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        aux_chans=20,
        use_spade=False,
        lambda_grad=0.1,
        lambda_water=1.0,
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        freeze_backbone=False,
        freeze_mode="none",
        spade_lr_scale=0.3,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.aux_chans       = aux_chans
        self.use_spade       = use_spade
        self.lambda_grad     = lambda_grad
        self.lambda_water    = lambda_water
        self.hr_mean         = hr_mean
        self.hr_std          = hr_std
        self.DATA_RANGE      = data_range
        self.DATA_MIN        = data_min
        self.freeze_backbone = freeze_backbone
        self.freeze_mode     = freeze_mode
        self.spade_lr_scale  = spade_lr_scale

        # ── BACKBONE ──────────────────────────────────────────
        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=1 + aux_chans,   # 21ch: will be expanded from 3ch pretrained          
            img_size=img_size,
            window_size=8,
            img_range=1.0,
            depths=[6] * 6,
            embed_dim=embed_dim,
            num_heads=[6] * 6,
            mlp_ratio=2.0,
            upsampler="pixelshuffle",
            resi_connection="1conv",
        )

        # ── SPLIT UPSAMPLE (before pretrained load) ───────────
        self.upsample_s1, self.upsample_s2 = _split_upsample(
            self.body.upsample
        )

        # ── OUTPUT HEAD ───────────────────────────────────────
        # conv_last outputs 3ch → squeeze to 1ch TIR
        self.out = nn.Conv2d(21, 1, 1)

        # ── TWO-STAGE SPADE ───────────────────────────────────
        # feat_ch=64 confirmed from checkpoint:
        #   conv_before_upsample: [64, 180, 3, 3] → 64ch out
        #   upsample stages preserve 64ch (256÷4=64 after pixelshuffle)
        if use_spade:
            self.spade_mid = SPADE(feat_ch=64, aux_ch=aux_chans)
            self.spade_hr  = SPADE(feat_ch=64, aux_ch=aux_chans)
        else:
            self.spade_mid = None
            self.spade_hr  = None

        # ── PRETRAINED LOAD (before input expansion) ──────────
        if pretrained_path:
            self._load_pretrained(pretrained_path)

        # ── EXPAND INPUT LAYER (after pretrained load) ────────
        self._expand_input_pretrained_mean()

        self._print_setup()

    # ──────────────────────────────────────────────────────────────
    def _expand_input_pretrained_mean(self):
        """
        Expand conv_first 3ch → 21ch.
        Average pretrained RGB filters into single-channel template,
        repeat for all 21 input channels.
        Preserves pretrained filter structure, equal init for all channels.
        """
        old      = self.body.conv_first
        in_chans = 1 + self.aux_chans   # 21

        new = nn.Conv2d(
            in_chans, old.out_channels,
            old.kernel_size, old.stride, old.padding,
            bias=(old.bias is not None),
        )
        with torch.no_grad():
            avg = old.weight.mean(dim=1, keepdim=True)   # [180, 1, 3, 3]
            new.weight[:] = avg.repeat(1, in_chans, 1, 1)
            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.body.conv_first = new
        print(f"[INFO] conv_first expanded 3 → {in_chans}ch (pretrained_mean init)")

    # ──────────────────────────────────────────────────────────────
    def _load_pretrained(self, path):
        ckpt    = torch.load(path, map_location="cpu", weights_only=True)
        state   = ckpt.get("params", ckpt)
        matched = {
            k: v for k, v in state.items()
            if k in self.body.state_dict()
            and v.shape == self.body.state_dict()[k].shape
        }
        missing = set(self.body.state_dict().keys()) - set(matched.keys())
        self.body.load_state_dict(matched, strict=False)
        print(
            f"[pretrained] loaded {len(matched)} keys, "
            f"skipped {len(missing)} (expected: conv_first shape mismatch)"
        )

    # ──────────────────────────────────────────────────────────────
    def _apply_freezing(self):
        if not self.freeze_backbone:
            return
        for p in self.body.parameters():
            p.requires_grad = False
        # body+first: keep conv_first trainable so expanded input
        # layer can adapt even when rest of backbone is frozen
        if self.freeze_mode == "body+first":
            for p in self.body.conv_first.parameters():
                p.requires_grad = True
        frozen = sum(1 for p in self.body.parameters() if not p.requires_grad)
        print(f"[INFO] Froze {frozen} backbone tensors (mode={self.freeze_mode})")

    # ──────────────────────────────────────────────────────────────
    def denormalize(self, x, mean=None, std=None):
        if mean is None or std is None:
            return x
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(std,  device=x.device, dtype=x.dtype)
        return x * std + mean

    # ──────────────────────────────────────────────────────────────
    def forward(self, batch):
        lr      = batch["lr"]       # [B,  1, 64,  64]
        aux_lr  = batch["aux_lr"]   # [B, 20, 64,  64]  direct input
        aux_mid = batch["aux_mid"]  # [B, 20, 128, 128] SPADE mid
        aux_hr  = batch["aux_hr"]   # [B, 20, 256, 256] SPADE hr

        # 1. Direct input fusion
        x = torch.cat([lr, aux_lr], dim=1)                     # [B, 21, 64, 64]

        # 2. Backbone feature extraction
        x_first = self.body.conv_first(x)                      # [B, 180, 64, 64]
        res      = self.body.forward_features(x_first)         # [B, 180, 64, 64]
        res      = self.body.conv_after_body(res) + x_first    # [B, 180, 64, 64] residual
        res      = self.body.conv_before_upsample(res)         # [B,  64, 64, 64]

        # 3. HQ reconstruction stage 1: 64 → 128px
        res = self.upsample_s1(res)                            # [B, 64, 128, 128]
        if self.spade_mid is not None:
            # aux_mid already 128×128 from patching — no interpolation
            res = self.spade_mid(res, aux_mid)

        # 4. HQ reconstruction stage 2: 128 → 256px
        res = self.upsample_s2(res)                            # [B, 64, 256, 256]
        if self.spade_hr is not None:
            # aux_hr already 256×256 from patching — no interpolation
            res = self.spade_hr(res, aux_hr)

        # 5. Output
        res = self.body.conv_last(res)                         # [B,  3, 256, 256]
        return self.out(res)                                   # [B,  1, 256, 256]

    # ──────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # ──────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        self._apply_freezing()
        params = []

        backbone_params = (
            [p for p in self.body.parameters() if p.requires_grad]
            if self.freeze_backbone
            else list(self.body.parameters())
        )
        if backbone_params:
            params.append({
                "params": backbone_params,
                "lr": self.hparams.learning_rate * 0.1,
            })

        params.append({
            "params": self.out.parameters(),
            "lr": self.hparams.learning_rate,
        })

        for module in [self.spade_mid, self.spade_hr]:
            if module is not None:
                params.append({
                    "params": module.parameters(),
                    "lr": self.hparams.learning_rate * self.spade_lr_scale,
                    "weight_decay": 1e-5,
                })

        opt = optim.Adam(params)
        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=6,
        )
        return {
            "optimizer": opt,
            "lr_scheduler": {"scheduler": sch, "monitor": "val_full_psnr"},
        }

    # ──────────────────────────────────────────────────────────────
    def _print_setup(self):
        n_body  = sum(p.numel() for p in self.body.parameters())
        n_spade = sum(
            p.numel()
            for m in [self.spade_mid, self.spade_hr]
            if m is not None
            for p in m.parameters()
        )
        print("\n========= SwinIR DIRECT + TWO-STAGE SPADE =========")
        print(f"  aux_chans      : {self.aux_chans}")
        print(f"  use_spade      : {self.use_spade}")
        print(f"  freeze_mode    : {self.freeze_mode}")
        print(f"  backbone params: {n_body:,}")
        print(f"  SPADE params   : {n_spade:,}")
        print("====================================================\n")