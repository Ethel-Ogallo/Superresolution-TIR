import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


# Projection module 
class AuxProjectionCNN(nn.Module):
    """
    Learns a fused representation of:
    TIR + AUX → 3-channel pseudo image
    """

    def __init__(self, in_chans, out_chans=3):
        super().__init__()

        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(32, out_chans, 1)
        )

    def forward(self, x):
        return self.net(x)


# SwinIR Lightning Module
class SwinIRModule(pl.LightningModule):

    def __init__(
        self,
        pretrained_path=None,
        learning_rate=1e-4,
        img_size=48,
        embed_dim=180,
        depths=None,
        num_heads=None,
        window_size=8,
        mlp_ratio=2.0,
        upsampler="pixelshuffle",
        data_range=None,
        hr_mean=None,
        hr_std=None,
        adaptation_strategy="projection",
        phase = 2,
    ):
        super().__init__()

        self.save_hyperparameters()

        depths = depths or [6]*6
        num_heads = num_heads or [6]*6

        self.DATA_RANGE = data_range
        self.hr_mean = hr_mean
        self.hr_std = hr_std

        self.adaptation_strategy = adaptation_strategy

        # lazy modules
        self.proj = None
        self._input_adapted = False

        # SwinIR backbone 
        self.body = swinir_arch.SwinIR(
            upscale=4,
            in_chans=3,
            img_size=img_size,
            window_size=window_size,
            img_range=1.0,
            depths=depths,
            embed_dim=embed_dim,
            num_heads=num_heads,
            mlp_ratio=mlp_ratio,
            upsampler=upsampler,
            resi_connection="1conv",
        )

        # NEW: output head 
        self.output_head = nn.Conv2d(
            in_channels=3,
            out_channels=1,
            kernel_size=1
        )

        # load pretrained AFTER architecture defined
        if pretrained_path:
            self._load_pretrained(pretrained_path)

    # ------- Input adaptation (projection only) --------
    def _adapt_input_layer(self, in_chans):

        old_conv = self.body.conv_first

        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        )

        with torch.no_grad():
            new_conv.weight[:, :3] = old_conv.weight

            if in_chans > 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                extra = in_chans - 3
                new_conv.weight[:, 3:] = mean_w.repeat(1, extra, 1, 1)

            new_conv.bias.copy_(old_conv.bias)

        self.body.conv_first = new_conv
        print(f"[INFO] conv_first adapted - {in_chans} channels")

    # ------------ Forward ------------
    def forward(self, batch):

        lr = batch["lr"]

        # baseline (no AUX)
        if "aux" not in batch or batch["aux"] is None:
            feat = self.body(lr)
            return self.output_head(feat)

        aux = batch["aux"]
        x = torch.cat([lr, aux], dim=1)

        # projection strategy
        if self.adaptation_strategy == "projection":

            if self.proj is None:
                self.proj = AuxProjectionCNN(x.shape[1], 3).to(x.device)
                print(f"[INFO] projection built {x.shape[1]} - 3")

            x = self.proj(x)

        elif self.adaptation_strategy == "direct":
            if not self._input_adapted:
                self._adapt_input_layer(x.shape[1])
                self._input_adapted = True

        else:
            raise ValueError("Unknown strategy")

        # SwinIR backbone
        feat = self.body(x)
        return self.output_head(feat)

    # Denormalization
    def denormalize(self, x):
        if self.hr_mean is None or self.hr_std is None:
            return x

        mean = torch.tensor(self.hr_mean, device=x.device, dtype=x.dtype)
        std = torch.tensor(self.hr_std, device=x.device, dtype=x.dtype)
        return x * std + mean

    # metrics
    def training_step(self, batch, batch_idx):
        return shared_step(self, batch, "train")

    def validation_step(self, batch, batch_idx):
        return shared_step(self, batch, "val")

    def test_step(self, batch, batch_idx):
        return shared_step(self, batch, "test")

    # -------- Optimizer --------
    def configure_optimizers(self):

        opt = optim.Adam(self.parameters(), lr=self.hparams.learning_rate)

        sch = optim.lr_scheduler.ReduceLROnPlateau(
            opt,
            mode="max",
            factor=0.5,
            patience=5
        )

        return {
            "optimizer": opt,
            "lr_scheduler": {
                "scheduler": sch,
                "monitor": "val_full_psnr"
            }
        }

    # ------- Pretrained loading --------
    def _load_pretrained(self, path):

        if not os.path.exists(path):
            print(f"[WARNING] missing {path}")
            return

        ckpt = torch.load(path, map_location="cpu", weights_only=True)
        state = ckpt.get("params", ckpt)

        model_dict = self.body.state_dict()

        matched = {
            k: v for k, v in state.items()
            if k in model_dict and v.shape == model_dict[k].shape
        }

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] loaded {len(matched)} SwinIR layers")