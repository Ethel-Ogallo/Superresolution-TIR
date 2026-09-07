import re
import time
import torch
import lightning.pytorch as pl
import torch.nn.functional as F
from basicsr.archs.basicvsrpp_arch import BasicVSRPlusPlus
from scripts.utils.seq_metrics import compute_metrics
from scripts.utils.loss import combined_loss
from scripts.models.spade import SPADEResnetBlock


# ----------- Pre-requisite functions -----------
def _remap_mmediting_key(key):
    m = re.match(r'^((?:spynet\.)?)basic_module\.(\d+)\.basic_module\.(\d+)\.conv\.(weight|bias)$', key)
    if m:
        prefix, b, i, wb = m.groups()
        return f"{prefix}basic_module.{b}.basic_module.{int(i) * 2}.{wb}"

    key = key.replace('upsample1.upsample_conv.', 'upconv1.')
    key = key.replace('upsample2.upsample_conv.', 'upconv2.')
    return key


def _load_with_fallbacks(module, state_dict, label):
    missing, unexpected = module.load_state_dict(state_dict, strict=False)

    if missing or unexpected:
        for prefix in ("generator.", "module.", "model."):
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
        print(f"[WARNING] {label}: {len(missing)} missing keys")
    if unexpected:
        print(f"[WARNING] {label}: {len(unexpected)} unexpected keys")
    if not missing and not unexpected:
        print(f"[INFO] {label} loaded cleanly.")

    return missing, unexpected


# ---------- BasicVSR++ with Auxiliary Data + SPADE ----------
class BasicVSRPlusPlusAux(BasicVSRPlusPlus):
    """
    BasicVSR++ with auxiliary input support.
    feat_extract receives [TIR + auxiliary channels].
    SPADE is injected in upsample(): landcover-only (seg_nc classes),
    at the two reconstruction resolutions (128x128, 256x256) 
    SPyNet unchanged: TIR is replicated to 3 channels before flow computation.
    """
    def __init__(self, *args, seg_nc=14, **kwargs):
        super().__init__(*args, **kwargs)
        mid_channels = kwargs.get("mid_channels", 64)
        self.spade_mid = SPADEResnetBlock(mid_channels, mid_channels, seg_nc)
        self.spade_hr  = SPADEResnetBlock(mid_channels, mid_channels, seg_nc)

    def forward(self, lqs, aux=None, aux_mid=None, aux_hr=None):
        # lqs:[B, T, 1, H, W]
        n, t, c, h, w = lqs.size()
        self.cpu_cache = True if t > self.cpu_cache_length else False

        # SPyNet branch
        if c == 1:
            lqs_flow_input = lqs.repeat(1, 1, 3, 1, 1)
        else:
            lqs_flow_input = lqs

        if self.is_low_res_input:
            lqs_downsample = lqs_flow_input.clone()
        else:
            lqs_downsample = F.interpolate(
                lqs_flow_input.view(-1, 3, h, w),
                scale_factor=0.25,
                mode='bicubic'
            ).view(n,t,3, h // 4, w // 4)

        self.check_if_mirror_extended(lqs_flow_input)

        # Feature extractor branch
        if aux is not None:
            feat_input = torch.cat( [lqs,aux], dim=2)  # aux:[B,T,Naux,H,W]
        else:
            feat_input = lqs  # baseline

        feats = {}

        if self.cpu_cache:
            feats['spatial'] = []
            for i in range(t):
                feat = self.feat_extract(feat_input[:, i]).cpu()
                feats['spatial'].append(feat)
                torch.cuda.empty_cache()
        else:
            feat_input = feat_input.view(-1,feat_input.shape[2],h, w)
            feats_ = self.feat_extract(feat_input)
            h_feat, w_feat = feats_.shape[2:]
            feats_ = feats_.view(n, t,-1, h_feat,w_feat)
            feats['spatial'] = [feats_[:, i] for i in range(t)]

        # Optical flow
        flows_forward, flows_backward = self.compute_flow(lqs_downsample)

        # Propagation
        for iter_ in [1, 2]:
            for direction in ['backward', 'forward']:
                module = f'{direction}_{iter_}'
                feats[module] = []
                if direction == 'backward':
                    flows = flows_backward
                elif flows_forward is not None:
                    flows = flows_forward
                else:
                    flows = flows_backward.flip(1)

                feats = self.propagate(feats,flows,module)

                if self.cpu_cache:
                    del flows
                    torch.cuda.empty_cache()

        # Reconstruction (SPADE-aware)
        return self.upsample(lqs_flow_input, feats, aux_mid, aux_hr)

    def upsample(self, lqs, feats, aux_mid=None, aux_hr=None):
        outputs = []
        num_outputs = len(feats['spatial'])
        mapping_idx = list(range(0, num_outputs))
        mapping_idx += mapping_idx[::-1]

        for i in range(lqs.size(1)):
            hr = [feats[k].pop(0) for k in feats if k != 'spatial']
            hr.insert(0, feats['spatial'][mapping_idx[i]])
            hr = torch.cat(hr, dim=1)

            hr = self.reconstruction(hr)
            hr = self.lrelu(self.pixel_shuffle(self.upconv1(hr)))
            if aux_mid is not None:
                hr = self.spade_mid(hr, aux_mid[:, i])       # 128x128 stage
            hr = self.lrelu(self.pixel_shuffle(self.upconv2(hr)))
            if aux_hr is not None:
                hr = self.spade_hr(hr, aux_hr[:, i])         # 256x256 stage
            hr = self.lrelu(self.conv_hr(hr))
            hr = self.conv_last(hr)

            if self.is_low_res_input:
                hr += self.img_upsample(lqs[:, i, :, :, :])
            else:
                hr += lqs[:, i, :, :, :]

            if self.cpu_cache:
                hr = hr.cpu()
                torch.cuda.empty_cache()

            outputs.append(hr)

        return torch.stack(outputs, dim=1)


# ----------------- BasicVSR Lightning Module -----------------
class BasicVSRModule(pl.LightningModule):

    def __init__(
        self,
        hr_mean=None,
        hr_std=None,
        data_range=None,
        data_min=None,
        mid_channels=64,
        num_blocks=7,
        max_residue_magnitude=10,
        is_low_res_input=True,
        cpu_cache_length=100,
        spynet_path=None,
        pretrained_path=None,
        use_aux=False,
        n_aux_channels=25,
        use_spade=False,
        seg_nc=14,
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
        self.use_aux = use_aux
        self.n_aux_channels = n_aux_channels
        self.use_spade = use_spade

        self.model = BasicVSRPlusPlusAux(
            mid_channels=mid_channels,
            num_blocks=num_blocks,
            max_residue_magnitude=max_residue_magnitude,
            is_low_res_input=is_low_res_input,
            spynet_path=None,
            cpu_cache_length=cpu_cache_length,
            seg_nc=seg_nc,
        )

        # Load pretrained SPyNet
        if spynet_path is not None:
            self.load_spynet(spynet_path)

        # Load pretrained BasicVSR++
        if pretrained_path is not None:
            self.load_pretrained(pretrained_path)

        # Expand feature extractor input channels
        self.expand_feature_extractor( n_aux_channels if use_aux else 0)

    def expand_feature_extractor(self, n_aux_channels):
        old_conv = self.model.feat_extract.main[0]
        out_channels, old_in_channels, kh, kw = old_conv.weight.shape
        new_in_channels = 1 + n_aux_channels

        new_conv = torch.nn.Conv2d(
            new_in_channels,
            out_channels,
            kernel_size=(kh, kw),
            stride=old_conv.stride,
            padding=old_conv.padding,
            bias=(old_conv.bias is not None)
        )

        with torch.no_grad():
            # Average pretrained RGB filters
            mean_weight = old_conv.weight.mean( dim=1, keepdim=True)
            new_conv.weight[:, 0:1] = mean_weight # Channel 0 = TIR

            # Channels 1:N = auxiliary
            if n_aux_channels > 0:
                new_conv.weight[:, 1:] = ( mean_weight.repeat(1, n_aux_channels, 1, 1))

            if old_conv.bias is not None:
                new_conv.bias.copy_(old_conv.bias)

        self.model.feat_extract.main[0] = new_conv

        print(f"[INFO] Feature extractor input channels: " f"{old_in_channels} -> {new_in_channels}" )

    # ----------- Forward -----------------
    def forward(self, lr_seq, aux_seq=None, aux_mid=None, aux_hr=None):
        if not self.use_aux:
            aux_seq = None
        if not self.use_spade:
            aux_mid = aux_hr = None
        return self.model(lr_seq, aux_seq, aux_mid, aux_hr)

    def flatten_for_loss(self, sr_seq, hr_seq, mask_hr_seq, mask_water_seq=None):
        """
        Flatten sequence tensors from [B, T, ...] to [B*T, ...]
        so losses and metrics can be computed frame-wise.
        """
        B, T = sr_seq.shape[:2]

        sr_flat = sr_seq[:, :, 0:1].reshape(B * T, 1, *sr_seq.shape[-2:])
        hr_flat = hr_seq.reshape(B * T, 1, *hr_seq.shape[-2:])
        m_hr_flat = mask_hr_seq.reshape(B * T, 1, *mask_hr_seq.shape[-2:])

        if mask_water_seq is not None:
            m_w_flat = mask_water_seq.reshape(B * T, 1, *mask_water_seq.shape[-2:])
        else:
            m_w_flat = None

        return sr_flat, hr_flat, m_hr_flat, m_w_flat

    # --------- Freeze SPyNet -------------
    def on_train_epoch_start(self):
        for param in self.model.spynet.parameters():
            param.requires_grad = False

    # ----------------- training ------------------------
    def training_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"], batch.get("aux"), batch.get("aux_mid"), batch.get("aux_hr"))
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch.get("water_mask")
        )

        time_gap = batch["time_gap"].reshape(-1)
        date_gap = batch["date_gap"].reshape(-1)

        loss_dict = combined_loss(
            self.denormalize(sr_flat),
            self.denormalize(hr_flat),
            m_hr,
            m_w,
            self.hparams.lambda_nw,
            self.hparams.lambda_w,
            self.hparams.lambda_g,
            time_gap_hours=time_gap,
            date_gap_days=date_gap,
        )

        self.log_dict({f"train/{k}": v for k, v in loss_dict.items()}, prog_bar=True)
        return loss_dict["loss_total"]

    # ---------------- validation -----------------
    @torch.no_grad()
    def validation_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"], batch.get("aux"), batch.get("aux_mid"), batch.get("aux_hr"))
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch.get("water_mask")
        )
        sr_denorm = self.denormalize(sr_flat)
        hr_denorm = self.denormalize(hr_flat)
        metrics = compute_metrics(self, sr_denorm, hr_denorm, m_hr, "val", m_w)
        val_logs = {f"val/{k}": v for k, v in metrics.items() if v is not None}
        self.log_dict(val_logs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    # ----------------- test ---------------
    @torch.no_grad()
    def test_step(self, batch, batch_idx):
        sr_seq = self(batch["lr"], batch.get("aux"), batch.get("aux_mid"), batch.get("aux_hr"))
        sr_flat, hr_flat, m_hr, m_w = self.flatten_for_loss(
            sr_seq, batch["hr"], batch["hr_mask"], batch.get("water_mask")
        )
        sr_denorm = self.denormalize(sr_flat)
        hr_denorm = self.denormalize(hr_flat)
        metrics = compute_metrics(self, sr_denorm, hr_denorm, m_hr, "test", m_w)
        test_logs = {f"test/{k}": v for k, v in metrics.items() if v is not None}
        self.log_dict(test_logs, on_step=False, on_epoch=True, prog_bar=True, sync_dist=True)

    # ---------------- optimizers ---------------
    def configure_optimizers(self):
        trainable_params = [p for p in self.parameters() if p.requires_grad]
        optimizer = torch.optim.Adam(trainable_params, lr=self.hparams.learning_rate)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5, min_lr=1e-6
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "monitor": "val/water_mae",
                "interval": "epoch",
            },
        }

    # -------------- Utilities --------------
    def denormalize(self, x):
        return x * self.hr_std + self.hr_mean

    def load_spynet(self, path):
        raw = torch.load(path, map_location="cpu")
        state_dict = raw.get("params") or raw.get("state_dict") or raw
        _load_with_fallbacks(self.model.spynet, state_dict, label="SpyNet")

    def load_pretrained(self, path):
        ckpt = torch.load(path, map_location="cpu")
        state_dict = ckpt.get("params") or ckpt.get("state_dict") or ckpt
        _load_with_fallbacks(self.model, state_dict, label="BasicVSR++")