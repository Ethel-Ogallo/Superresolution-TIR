"""
basicvsrplus.py — BasicVSR++ Lightning Module for TIR Sequential Super-Resolution

"""

import re
import torch
import lightning.pytorch as pl
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from scripts.utils.metrics import compute_metrics
from scripts.utils.loss import combined_loss


def _remap_mmediting_key(key):
    """Maps a single mmediting/mmagic-style BasicVSR++ key to basicsr's expected
    name. Handles two known structural differences (confirmed against an actual
    checkpoint, not guessed):
      1. SpyNet: mmediting wraps each conv stage in a ConvModule
         ('basic_module.B.basic_module.I.conv.W'), basicsr stores Conv2d
         directly inside a Sequential alongside ReLU, at even positions
         ('basic_module.B.basic_module.2I.W'). Applies whether or not the key
         has a leading 'spynet.' (standalone SpyNet ckpt vs. embedded in the
         full backbone ckpt).
      2. Upsampling: mmediting names these 'upsample1.upsample_conv' /
         'upsample2.upsample_conv', basicsr names them 'upconv1' / 'upconv2'.
    """
    m = re.match(r'^((?:spynet\.)?)basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)$', key)
    if m:
        prefix, b, i, wb = m.groups()
        return f"{prefix}basic_module.{b}.basic_module.{int(i) * 2}.{wb}"

    key = key.replace('upsample1.upsample_conv.', 'upconv1.')
    key = key.replace('upsample2.upsample_conv.', 'upconv2.')
    return key


def _load_with_fallbacks(module, state_dict, label):
    """Shared loader: tries a direct load, then common prefix stripping, then
    the mmediting key remap -- in that order, keeping whichever attempt leaves
    the fewest missing keys. Always reports the final missing/unexpected count
    so a silent partial load never goes unnoticed."""
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        for prefix in ("generator.", "module.", "net_g."):
            if any(k.startswith(prefix) for k in state_dict.keys()):
                stripped = {k[len(prefix):]: v for k, v in state_dict.items() if k.startswith(prefix)}
                m2, u2 = module.load_state_dict(stripped, strict=False)
                if len(m2) < len(missing):
                    missing, unexpected, state_dict = m2, u2, stripped
                    print(f"[INFO] {label}: loaded after stripping '{prefix}' prefix.")
                break

    if missing or unexpected:
        remapped = {_remap_mmediting_key(k): v for k, v in state_dict.items()}
        m2, u2 = module.load_state_dict(remapped, strict=False)
        if len(m2) < len(missing):
            missing, unexpected = m2, u2
            print(f"[INFO] {label}: loaded after remapping mmediting key layout.")

    if missing:
        print(f"[WARNING] {label}: {len(missing)} missing keys (first 5: {missing[:5]})")
    if unexpected:
        print(f"[WARNING] {label}: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")
    if not missing and not unexpected:
        print(f"[INFO] {label} loaded cleanly, no missing/unexpected keys.")
    return missing, unexpected


class BasicVSRModule(pl.LightningModule):
    def __init__(
        self,
        # --- normalization stats (from stats.json, passed in at construction) ---
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        # --- architecture (defaults match the c64n7 pretrained checkpoint) ---
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        cpu_cache_length=100,
        spynet_path=None,
        # --- pretrained backbone (full net_g checkpoint, loaded post-construction) ---
        pretrained_path=None,
        # --- loss weights ---
        lambda_nw=1.0,
        lambda_w=2.0,
        lambda_g=0.1,
        learning_rate=1e-4,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()

        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.DATA_RANGE = data_range
        self.DATA_MIN = data_min

        # ------- Architecture: BasicVSR++ (verified basicsr signature) -------
        # NOTE: we do NOT pass spynet_path into the constructor. basicsr's own
        # SpyNet.__init__ hardcodes `torch.load(load_path)['params']` with no
        # fallback -- if your checkpoint isn't wrapped exactly that way (e.g.
        # it's an mmediting-style {'state_dict': ...} checkpoint, or a bare
        # state dict with no wrapper), it crashes with KeyError('params')
        # before the model even finishes constructing. We load it ourselves
        # below instead, defensively, the same way we already handle the
        # backbone checkpoint.
        self.net_g = BasicVSRPlusPlus(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
            spynet_path=None,
            cpu_cache_length=cpu_cache_length,
        )

        if spynet_path is not None:
            self.load_spynet(spynet_path)

        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

    # ---------------- forward ----------------
    def forward(self, lr_seq):
        # lr_seq: [B, T, 3, H, W] -> BasicVSR++ returns [B, T, 3, H, W]
        return self.net_g(lr_seq)

    # ---------------- flatten bridge (shared by train/val/test) ----------------
    def flatten_for_loss(self, sr_seq, hr_seq, mask_hr, mask_water):
        """[B, T, C, H, W] -> [B*T, 1, H, W]. Masks already carry a per-frame T
        axis (each frame is a different downstream tile), so they're reshaped
        directly -- NOT repeat_interleaved, which would double-count them."""
        B, T, C, H, W = sr_seq.shape
        sr_flat = sr_seq[:, :, 0:1, :, :].reshape(B * T, 1, H, W)
        hr_flat = hr_seq[:, :, 0:1, :, :].reshape(B * T, 1, H, W)
        m_hr = mask_hr.reshape(B * T, 1, H, W)
        m_w = mask_water.reshape(B * T, 1, H, W)
        return sr_flat, hr_flat, m_hr, m_w

    # ---------------- training_step ----------------
    def training_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"])
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch["water_mask"]
        )

        T = sr_seq.shape[1]
        time_gap = batch["time_gap"].repeat_interleave(T)   # scalar per sequence -> per frame
        date_gap = batch["date_gap"].repeat_interleave(T)

        loss_dict = combined_loss(
            self.denormalize(sr_flat), self.denormalize(hr_flat),
            m_hr, m_w,
            self.hparams.lambda_nw, self.hparams.lambda_w, self.hparams.lambda_g,
            time_gap_hours=time_gap, date_gap_days=date_gap,
        )
        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        return loss_dict["loss_total"]

    # ---------------- validation_step ----------------
    def validation_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"])
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch["water_mask"]
        )
        sr_denorm = self.denormalize(sr_flat)
        hr_denorm = self.denormalize(hr_flat)
        metrics = compute_metrics(self, sr_denorm, hr_denorm, m_hr, "val", m_w)
        self.log_dict({f"val/{k}": v for k, v in metrics.items() if v is not None})

    # ---------------- test_step ----------------
    def test_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"])
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch["water_mask"]
        )
        sr_denorm = self.denormalize(sr_flat)
        hr_denorm = self.denormalize(hr_flat)
        metrics = compute_metrics(self, sr_denorm, hr_denorm, m_hr, "test", m_w)
        self.log_dict({f"test/{k}": v for k, v in metrics.items() if v is not None})

    # ---------------- optimizer ----------------
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/water_mae",   # same metric your checkpoint/early-stop already track
                "interval": "epoch",
                "frequency": 1,
            },
        }

    # ---------------- helpers ----------------
    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean

    def load_spynet(self, path):
        """Loads SpyNet weights. The 'mean'/'std' keys will always show as
        'missing' -- those are fixed constants registered in code (ImageNet
        normalization), not learned weights, and are never in any checkpoint.
        That specific warning is expected and harmless."""
        raw = torch.load(path, map_location="cpu")
        if isinstance(raw, dict) and "params" in raw:
            state_dict = raw["params"]
        elif isinstance(raw, dict) and "state_dict" in raw:
            state_dict = raw["state_dict"]
        elif isinstance(raw, dict):
            state_dict = raw
        else:
            raise RuntimeError(f"Unrecognized SpyNet checkpoint format at {path}: {type(raw)}")

        _load_with_fallbacks(self.net_g.spynet, state_dict, label="SpyNet")

    def load_pretrained(self, path):
        """Loads the full BasicVSR++ backbone checkpoint (SpyNet's own weights
        are loaded separately via load_spynet, called earlier in __init__).
        Applies the same prefix-stripping and mmediting-key-remap fallbacks,
        since the embedded spynet weights and upsample layer names inside this
        checkpoint use mmediting's naming too, not just the top-level wrapper."""
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("params", ckpt.get("state_dict", ckpt))
        _load_with_fallbacks(self.net_g, state_dict, label="BasicVSR++ backbone")