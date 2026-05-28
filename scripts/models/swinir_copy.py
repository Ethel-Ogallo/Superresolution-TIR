"""
SwinIR with:
  - Direct input fusion: LR TIR (1ch) + aux_lr (20ch) → 21ch into conv_first
  - Pretrained-mean weight init for expanded input layer
  - Two-stage True SPADE in HQ reconstruction block (Direct Aux guidance feeding)
"""
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl
from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


# =========================================================
# OFFICIAL SPEC SPADE BLOCK
# =========================================================
class SPADE(nn.Module):
    def __init__(self, feat_ch, aux_ch, hidden=128):
        super().__init__()
        # The official small-batch fallback choice
        self.norm = nn.InstanceNorm2d(feat_ch, affine=False)

        self.shared = nn.Sequential(
            nn.Conv2d(aux_ch, hidden, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )
        self.gamma = nn.Conv2d(hidden, feat_ch, kernel_size=3, padding=1)
        self.beta = nn.Conv2d(hidden, feat_ch, kernel_size=3, padding=1)

    def forward(self, x, aux):
        x_norm = self.norm(x)
        h = self.shared(aux)
        return x_norm * (1 + self.gamma(h)) + self.beta(h)


# =========================================================
# SPLIT UPSAMPLE
# =========================================================
def _split_upsample(upsample_seq):
    layers = list(upsample_seq.children())
    assert len(layers) == 4, f"Expected 4 layers, got {len(layers)}"
    stage1 = nn.Sequential(layers[0], layers[1])
    stage2 = nn.Sequential(layers[2], layers[3])
    return stage1, stage2


# =========================================================
# MAIN MODEL
# =========================================================
class SwinIRModule(pl.LightningModule):
    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        aux_chans=20,
        use_spade=True,
        lambda_grad=0.0,
        lambda_water=0.0,
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        freeze_backbone=False,
        freeze_mode="none",
        spade_lr_scale=0.3,      # Lower LR for SPADE stability
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.lambda_grad = lambda_grad
        self.lambda_water = lambda_water
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE   = data_range
        self.DATA_MIN     = data_min

        self.aux_chans = aux_chans
        self.use_spade = use_spade
        self.freeze_backbone = freeze_backbone
        self.freeze_mode = freeze_mode
        self.spade_lr_scale = spade_lr_scale

        # Backbone
        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=1 + aux_chans,
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

        self.upsample_s1, self.upsample_s2 = _split_upsample(self.body.upsample)

        self.final_proj = nn.Conv2d(21, 3, kernel_size=1)
        self.out = nn.Conv2d(3, 1, kernel_size=1)

        self.feat_ch_for_spade = 64

        if use_spade:
            # SPADE now takes raw auxiliary maps (aux_chans) directly without an AuxEncoder
            self.spade_mid = SPADE(feat_ch=self.feat_ch_for_spade, aux_ch=aux_chans)
            self.spade_hr = SPADE(feat_ch=self.feat_ch_for_spade, aux_ch=aux_chans)
        else:
            self.spade_mid = None
            self.spade_hr = None

        if pretrained_path:
            self._load_pretrained(pretrained_path)

        self._expand_input_pretrained_mean()
        self._print_setup()

    # ──────────────────────────────────────────────────────────────
    # INPUT EXPANSION
    # ──────────────────────────────────────────────────────────────
    def _expand_input_pretrained_mean(self):
        old      = self.body.conv_first
        in_chans = 1 + self.aux_chans   # 21

        new = nn.Conv2d(
            in_chans,
            old.out_channels,
            old.kernel_size,
            old.stride,
            old.padding,
            bias=(old.bias is not None),
        )

        with torch.no_grad():
            avg = old.weight.mean(dim=1, keepdim=True)
            new.weight[:] = avg.repeat(1, in_chans, 1, 1)
            if old.bias is not None:
                new.bias.copy_(old.bias)

        self.body.conv_first = new
        print(f"[INFO] conv_first expanded 3 → {in_chans}ch (pretrained_mean init)")

    # ──────────────────────────────────────────────────────────────
    # PRETRAINED LOAD
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
        print(f"[pretrained] loaded {len(matched)} keys, skipped {len(missing)} (shape mismatch)")

    # ──────────────────────────────────────────────────────────────
    # FREEZING
    # ──────────────────────────────────────────────────────────────
    def _apply_freezing(self):
        if not self.freeze_backbone:
            return

        for p in self.body.parameters():
            p.requires_grad = False

        if self.freeze_mode == "body+first":
            for p in self.body.conv_first.parameters():
                p.requires_grad = True

        frozen = sum(1 for p in self.body.parameters() if not p.requires_grad)
        print(f"[INFO] Froze {frozen} backbone parameter tensors (mode={self.freeze_mode})")

    # ──────────────────────────────────────────────────────────────
    # DENORMALIZE
    # ──────────────────────────────────────────────────────────────
    def denormalize(self, x, mean=None, std=None):
        if mean is None or std is None:
            return x
        mean = torch.tensor(mean, device=x.device, dtype=x.dtype)
        std  = torch.tensor(std,  device=x.device, dtype=x.dtype)
        return x * std + mean

    # ──────────────────────────────────────────────────────────────
    # FORWARD
    # ──────────────────────────────────────────────────────────────
    def forward(self, batch):
        lr = batch["lr"]
        aux_lr = batch["aux_lr"]
        aux_mid = batch["aux_mid"]
        aux_hr = batch["aux_hr"]

        x = torch.cat([lr, aux_lr], dim=1)                    # [B, 21, 64, 64]

        feat = self.body.conv_first(x)
        feat = self.body.forward_features(feat)
        feat = self.body.conv_before_upsample(feat)

        # Upsample + SPADE Stage 1 (Direct raw guidance passing)
        feat = self.upsample_s1(feat)                         # [B, 64, 128, 128]
        if self.spade_mid is not None:
            feat = self.spade_mid(feat, aux_mid)

        # Upsample + SPADE Stage 2 (Direct raw guidance passing)
        feat = self.upsample_s2(feat)                         # [B, 64, 256, 256]
        if self.spade_hr is not None:
            feat = self.spade_hr(feat, aux_hr)

        # SwinIR final layer (should naturally output 21 channels)
        feat = self.body.conv_last(feat)                      # [B, 21, 256, 256]

        # Collapse to target 1 channel
        feat = self.final_proj(feat)                          # [B, 3, 256, 256]
        return self.out(feat)                                 # [B, 1, 256, 256]

    # ──────────────────────────────────────────────────────────────
    # LIGHTNING STEPS
    # ──────────────────────────────────────────────────────────────
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # ──────────────────────────────────────────────────────────────
    # OPTIMIZER + SCHEDULER
    # ──────────────────────────────────────────────────────────────
    def configure_optimizers(self):
        self._apply_freezing()
        params = []

        # Backbone — very conservative LR
        backbone_params = [p for p in self.body.parameters() if p.requires_grad]
        if backbone_params:
            params.append({
                "params": backbone_params,
                "lr": self.hparams.learning_rate * 0.1,
            })

        # Output head
        params.append({
            "params": self.out.parameters(),
            "lr": self.hparams.learning_rate,
        })

        # SPADE blocks — Modified to strip out AuxEncoder completely
        for module in [self.spade_mid, self.spade_hr]:
            if module is not None:
                params.append({
                    "params": module.parameters(),
                    "lr": self.hparams.learning_rate * self.hparams.spade_lr_scale,
                    "weight_decay": 1e-5,
                })

        opt = optim.Adam(params)

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt, mode="max", factor=0.5, patience=6, verbose=True
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_full_psnr",
            },
        }

    # ──────────────────────────────────────────────────────────────
    def _print_setup(self):
        n_body = sum(p.numel() for p in self.body.parameters())
        n_spade = sum(
            p.numel()
            for m in [self.spade_mid, self.spade_hr]
            if m is not None
            for p in m.parameters()
        )
        print("\n========= SwinIR DIRECT + TWO-STAGE TRUE SPADE =========")
        print(f" aux_chans       : {self.aux_chans}")
        print(f" use_spade       : {self.use_spade}")
        print(f" spade_lr_scale  : {self.spade_lr_scale}")
        print(f" freeze_mode     : {self.freeze_mode}")
        print(f" backbone params : {n_body:,}")
        print(f" SPADE params    : {n_spade:,}")
        print("====================================================\n")