import json
import sys
from pathlib import Path

import torch

# --------------------------------------------------
# Paths
# --------------------------------------------------

SISR_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = SISR_ROOT.parent

FASTDIFFSR_ROOT = (
    SISR_ROOT / "scripts" / "external" / "FastDiffSR"
)

CONFIG_PATH = (
    FASTDIFFSR_ROOT / "config" / "sr_fastdiffsr_test_64_256.json"
)

CHECKPOINT_PATH = (
    PROJECT_ROOT / "data" / "pretrained" / "fastdiffx4.pth"
)

sys.path.insert(0, str(SISR_ROOT))
sys.path.insert(0, str(FASTDIFFSR_ROOT.parent))

from FastDiffSR import networks

from scripts.models.spade_diff import FastDiffSRSPADEWrapper
from scripts.models.fastdiffsr_spade import (
    load_pretrained_weights,
    expand_input_conv,
)

# --------------------------------------------------
# Build original FastDiffSR
# --------------------------------------------------

with open(CONFIG_PATH, "r") as f:
    config = json.load(f)

config["distributed"] = False

netG = networks.define_G(config)

netG.set_new_noise_schedule(
    config["model"]["beta_schedule"]["train"],
    device=torch.device("cpu"),
)

netG.set_loss(torch.device("cpu"))

load_pretrained_weights(netG, CHECKPOINT_PATH)

# Expand original input convolution: 6 -> 31 channels.
first_conv = netG.denoise_fn.downs[0]

netG.denoise_fn.downs[0] = expand_input_conv(
    first_conv,
    new_in_channels=31,
)

# --------------------------------------------------
# Add SPADE wrapper
# --------------------------------------------------

netG.denoise_fn = FastDiffSRSPADEWrapper(
    unet=netG.denoise_fn,
    aux_channels=23,
    label_nc=14,
    label_start=8,
)

netG.eval()

# --------------------------------------------------
# Dummy inputs
# --------------------------------------------------

batch_size = 1

denoiser_input = torch.randn(batch_size, 31, 256, 256)
time = torch.rand(batch_size, 1)

aux_lr = torch.randn(batch_size, 23, 64, 64)
aux_mid = torch.randn(batch_size, 23, 128, 128)
aux_hr = torch.randn(batch_size, 23, 256, 256)

# --------------------------------------------------
# Forward test
# --------------------------------------------------

with torch.no_grad():

    output = netG.denoise_fn(
        denoiser_input,
        time,
        aux_lr,
        aux_mid,
        aux_hr,
    )

print("\nForward test completed.")
print("Input shape:", denoiser_input.shape)
print("Output shape:", output.shape)

assert output.shape == (batch_size, 3, 256, 256)

print("Output shape check: PASSED")

# --------------------------------------------------
# Trainable SPADE parameters
# --------------------------------------------------

spade_parameters = [
    (name, parameter)
    for name, parameter in netG.denoise_fn.named_parameters()
    if "spade_layers" in name
]

print("\nSPADE parameter groups:")

for name, parameter in spade_parameters:
    print(name, parameter.shape, "requires_grad:", parameter.requires_grad)

assert len(spade_parameters) > 0
assert all(parameter.requires_grad for _, parameter in spade_parameters)

print("\nSPADE parameter registration check: PASSED")