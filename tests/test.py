import importlib.util
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

# ---------------------------------------------------------
# 1. Locate the repository and load the existing train.py
# ---------------------------------------------------------
ROOT = Path.cwd().resolve()
TRAIN_FILE = ROOT / "SISR" / "scripts" / "training" / "01_train.py"

if not TRAIN_FILE.exists():
    raise FileNotFoundError(f"Training script not found: {TRAIN_FILE}")

sys.path.insert(0, str(ROOT / "SISR"))

spec = importlib.util.spec_from_file_location("training_module", TRAIN_FILE)
training_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(training_module)

SRDataset = training_module.SRDataset
PATCHES_DIR = training_module.PATCHES_DIR
STATS_PATH = training_module.STATS_PATH
PRETRAINED = training_module.PRETRAINED
PRETRAINED_D_PATHS = training_module.PRETRAINED_D_PATHS

from scripts.models.fastdiffsr import FastDiffSRModule

# ---------------------------------------------------------
# 2. Check the hardware
# ---------------------------------------------------------
if not torch.cuda.is_available():
    raise RuntimeError(
        "CUDA is unavailable. Run this test on the GPU node "
        "or in an environment with a working CUDA-enabled PyTorch."
    )

device = torch.device("cuda")
print(f"[INFO] GPU: {torch.cuda.get_device_name(0)}")
print(f"[INFO] Stats: {STATS_PATH}")
print(f"[INFO] Patches: {PATCHES_DIR}")

# ---------------------------------------------------------
# 3. Load one batch from the EXISTING training split
#    No changes to the dataset or preprocessing.
# ---------------------------------------------------------
train_ds = SRDataset(
    split="train",
    patches_dir=PATCHES_DIR,
    stats_path=STATS_PATH,
    repeat_channels=True,
    transform=None,
    use_water_mask=True,
)

if len(train_ds) == 0:
    raise RuntimeError("The training dataset is empty.")

loader = DataLoader(
    train_ds,
    batch_size=1,
    shuffle=False,
    num_workers=0,
)

batch = next(iter(loader))

print(f"[INFO] Training patches: {len(train_ds)}")
for key, value in batch.items():
    if isinstance(value, torch.Tensor):
        print(f"[BATCH] {key}: shape={tuple(value.shape)}, dtype={value.dtype}")

# ---------------------------------------------------------
# 4. Load FastDiffSR with the pretrained x4 generator
# ---------------------------------------------------------
checkpoint_path = PRETRAINED / PRETRAINED_PATHS["fastdiffsr"]

if not checkpoint_path.exists():
    raise FileNotFoundError(f"Checkpoint not found: {checkpoint_path}")

model = FastDiffSRModule(
    stats_path=str(STATS_PATH),
    learning_rate=1e-5,
    weight_decay=1e-6,
    checkpoint_path=str(checkpoint_path),
).to(device)

model.train()

batch = {
    key: value.to(device) if isinstance(value, torch.Tensor) else value
    for key, value in batch.items()
}

# ---------------------------------------------------------
# 5. Prepare the batch and calculate the diffusion loss
# ---------------------------------------------------------
optimizer = torch.optim.Adam(
    model.netG.parameters(),
    lr=1e-5,
    weight_decay=1e-6,
)

optimizer.zero_grad(set_to_none=True)

lr_up, hr, hr_mask = model._prepare_batch(batch)

print(f"[INFO] LR upsampled: {tuple(lr_up.shape)}")
print(f"[INFO] HR target:   {tuple(hr.shape)}")
print(f"[INFO] HR mask:     {tuple(hr_mask.shape)}")
print(f"[INFO] Valid HR pixels: {int((hr_mask > 0.5).sum())}")

if not (hr_mask > 0.5).any():
    raise RuntimeError("This batch contains no valid HR pixels.")

loss = model._masked_diffusion_loss(hr, lr_up, hr_mask)

print(f"[RESULT] Diffusion loss: {loss.item():.6f}")

if not torch.isfinite(loss):
    raise RuntimeError(f"Non-finite diffusion loss: {loss.item()}")

# ---------------------------------------------------------
# 6. Backpropagate and inspect gradients
# ---------------------------------------------------------
loss.backward()

parameters_with_grad = 0
parameters_with_nonzero_grad = 0
total_grad_sq = 0.0
parameter_to_check = None
parameter_before = None

for name, parameter in model.netG.named_parameters():
    if not parameter.requires_grad or parameter.grad is None:
        continue

    parameters_with_grad += 1
    grad = parameter.grad.detach()

    if not torch.isfinite(grad).all():
        raise RuntimeError(f"Non-finite gradient in {name}")

    grad_norm = grad.norm().item()
    total_grad_sq += grad_norm ** 2

    if grad_norm > 0:
        parameters_with_nonzero_grad += 1

        if parameter_to_check is None:
            parameter_to_check = (name, parameter)
            parameter_before = parameter.detach().clone()

total_grad_norm = total_grad_sq ** 0.5

print(f"[RESULT] Parameters with gradients: {parameters_with_grad}")
print(f"[RESULT] Parameters with nonzero gradients: {parameters_with_nonzero_grad}")
print(f"[RESULT] Total gradient norm: {total_grad_norm:.6f}")

if parameters_with_grad == 0 or parameters_with_nonzero_grad == 0:
    raise RuntimeError("No useful gradients reached the model.")

# ---------------------------------------------------------
# 7. Perform one optimizer update and verify a change
# ---------------------------------------------------------
optimizer.step()

name, parameter = parameter_to_check
parameter_change = (parameter.detach() - parameter_before).abs().max().item()

print(f"[RESULT] Checked parameter: {name}")
print(f"[RESULT] Maximum parameter change: {parameter_change:.10e}")

if not parameter_change > 0:
    raise RuntimeError("The optimizer step did not change the checked parameter.")

print("\n[SUCCESS] One-batch FastDiffSR gradient/update test passed.")
print("[NOTE] This was one optimizer update, not a full training run.")
