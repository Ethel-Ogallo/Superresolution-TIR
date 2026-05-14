import os
import torch
import torch.nn as nn
import torch.optim as optim
import lightning.pytorch as pl

from basicsr.archs import swinir_arch
from scripts.utils.metrics import shared_step


# Projection module 
class AuxProjection(nn.Module):
    def __init__(self, in_chans, out_chans=3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_chans, 64, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64, 32, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(32, out_chans, 1)
        )
        # projects TIR from 1 → 3 channels to match output shape
        self.tir_skip = nn.Conv2d(1, out_chans, kernel_size=1)

    def forward(self, x):
        tir = x[:, 0:1, :, :]       # extract TIR channel
        fused = self.net(x)          # full 21-channel fusion
        skip = self.tir_skip(tir)    # TIR projected to 3 channels
        return fused + skip          # TIR always contributes


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
        in_aux_chans=None,
        lambda_grad=0.0,   
        lambda_water=0.0,                
    ):
        super().__init__()
        self.save_hyperparameters()

        depths = depths or [6]*6
        num_heads = num_heads or [6]*6

        self.DATA_RANGE = data_range
        self.hr_mean = hr_mean
        self.hr_std = hr_std
        self.adaptation_strategy = adaptation_strategy
        self._input_adapted = False
        self.lambda_grad = lambda_grad
        self.lambda_water = lambda_water

        # Register proj properly — only if aux is being used with projection strategy
        if adaptation_strategy == "projection" and in_aux_chans is not None:
            self.proj = AuxProjection(1 + in_aux_chans, 3)  # 1 TIR + aux chans
        else:
            self.proj = None

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

        self.output_head = nn.Conv2d(3, 1, kernel_size=1)

        if pretrained_path:
            self._load_pretrained(pretrained_path)
        
        if adaptation_strategy == "direct" and in_aux_chans is not None:
            self._adapt_input_layer(1 + in_aux_chans)
            self._input_adapted = True

    # ------- Input adaptation --------
    def _adapt_input_layer(self, in_chans):

        old_conv = self.body.conv_first

        new_conv = nn.Conv2d(
            in_chans,
            old_conv.out_channels,
            kernel_size=old_conv.kernel_size,
            stride=old_conv.stride,
            padding=old_conv.padding,
        ).to(
            device=old_conv.weight.device,
            dtype=old_conv.weight.dtype
        )

        with torch.no_grad():

            new_conv.weight[:, :3] = old_conv.weight

            if in_chans > 3:
                mean_w = old_conv.weight.mean(dim=1, keepdim=True)
                extra = in_chans - 3
                new_conv.weight[:, 3:] = mean_w.repeat(1, extra, 1, 1)

            new_conv.bias.copy_(old_conv.bias)

        self.body.conv_first = new_conv

        self.body.mean = torch.tensor(
            0.0,
            device=old_conv.weight.device,
            dtype=old_conv.weight.dtype
        )

        self.body.img_range = 1.0

        print(f"[INFO] conv_first adapted - {in_chans} channels")

    # ------------ Forward ------------
    def forward(self, batch):
        lr = batch["lr"]

        if "aux" not in batch or batch["aux"] is None:
            feat = self.body(lr)
            return self.output_head(feat)

        aux = batch["aux"]
        x = torch.cat([lr, aux], dim=1)

        if self.adaptation_strategy == "projection":
            x = self.proj(x) 

        elif self.adaptation_strategy == "direct":
            pass  

        else:
            raise ValueError("Unknown strategy")

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

        # separate parameter groups with different LRs
        param_groups = [
            {
                "params": self.body.parameters(),  # pretrained backbone
                "lr": self.hparams.learning_rate * 0.1  # 10x smaller
            },
            {
                "params": self.output_head.parameters(),  # new head
                "lr": self.hparams.learning_rate
            },
        ]

        # only add proj group if it exists
        if self.proj is not None:
            param_groups.append({
                "params": self.proj.parameters(),  # randomly initialized
                "lr": self.hparams.learning_rate
            })

        opt = optim.Adam(param_groups)

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
        unmatched = [k for k in state if k not in matched]  # ADD THIS

        self.body.load_state_dict(matched, strict=False)
        print(f"[INFO] loaded {len(matched)} SwinIR layers")
        print(f"[INFO] {len(unmatched)} keys not loaded: {unmatched[:5]}")  # ADD THIS