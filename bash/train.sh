#!/bin/bash -l
#SBATCH --job-name=sisr_edsr            # Job name
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-gpu=6                   # 4 CPU cores
#SBATCH --mem=24G                             # Memory per node

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p results/logs/edsr

# -------- Run training --------
# python -m scripts.training.train \
#     --model edsr \
#     --metadata_json data/full_metadata.json \
#     --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth \
#     --random_split \
#     --lr 1e-4 \
#     --lambda_grad 0.1 \
#     --bb_lr_scale 0.1 \
#     --max_epochs 100 \
#     --patience 30 \
#     --batch_size 2  \
#     --num_workers 2 \
#     --run_name experiment1_v3 \
#     --group prelim \
#     --project TIR_sisr 
#     # --freeze_backbone 
python -m scripts.training.train \
    --model swinir \
    --metadata_json data/full_metadata.json \
    --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/SwinIR_Large_x4.pth \
    --random_split \
    --lr 1e-4 \
    --lambda_grad 0.1 \
    --bb_lr_scale 0.1 \
    --max_epochs 100 \
    --patience 30 \
    --batch_size 2 \
    --num_workers 2 \
    --run_name experiment2_full \
    --group prelim \
    --project TIR_sisr 
    # --freeze_backbone

# EDSR: /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth
# SwinIR: /share/home/e2406751/Superresolution-TIR/data/pretrained/SwinIR_Large_x4.pth

# ===============================
# SLURM + BATCH SIZE GUIDELINES
# ===============================

# DATA / MODEL SETUP
# -------------------------------
# HR size: 512 x 512
# LR size: 128 x 128
# Scale factor: x4
# Model patch size: 48 x 48 (LR) → 192 x 192 (HR)

# Each LR image (128x128) is internally split into ~4–9 patches
# → This MULTIPLIES the effective batch size

# Effective batch size:
# effective_batch = batch_size × patches_per_image
# Example:
# batch_size = 4 → effective_batch ≈ 16–36


# ===============================
# BATCH SIZE RECOMMENDATIONS
# ===============================

# batch_size = 2
# - Very safe
# - Low GPU usage
# - Slower training

# batch_size = 4  ✅ RECOMMENDED
# - Best balance
# - Stable for most GPUs
# - Effective batch ≈ 16–36

# batch_size = 8
# - Only if GPU ≥ 16GB VRAM
# - Monitor with: nvidia-smi
# - Risk of CUDA OOM

# batch_size = 16
# - Not recommended
# - Very likely to run out of memory


# ===============================
# SLURM SETTINGS GUIDELINES
# ===============================

# GPU (main bottleneck)
# -------------------------------
# batch ↑ → GPU memory ↑
# patch size ↑ → GPU memory ↑↑ (very strong effect)

# CPUs (data loading / rasterio)
# -------------------------------
# batch_size 2–4  → 4–6 CPUs
# batch_size 8    → 6–8 CPUs
# batch_size 16   → 8–12 CPUs

# RAM (GeoTIFF + preprocessing)
# -------------------------------
# batch_size 2–4  → 16–24 GB
# batch_size 8    → 24–32 GB
# batch_size 16   → 32–64 GB


# ===============================
# RECOMMENDED SLURM CONFIGS
# ===============================

# ✅ SAFE DEFAULT (batch_size = 4)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=24G


# 🚀 HIGHER BATCH (batch_size = 8)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=8
#SBATCH --mem=32G


# ⚠️ AGGRESSIVE (batch_size = 16 - risky)
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=12
#SBATCH --mem=48G


# ===============================
# IMPORTANT NOTES
# ===============================

# - Dataset size DOES NOT impact GPU memory much
# - Batch size DOES impact GPU memory directly
# - Internal patching is the main hidden cost

# Always monitor GPU usage:
#   nvidia-smi

# If you hit CUDA OOM:
#   → reduce batch size (first action)


# ===============================
# BEST PRACTICE (RECOMMENDED)
# ===============================

# Instead of increasing batch size:
# Use gradient accumulation

# Example:
# batch_size = 4
# accumulation_steps = 4
# → effective batch size = 16

# Benefits:
# - Same training effect
# - Much lower memory usage
# - More stable


# ===============================
# QUICK RULE OF THUMB
# ===============================

# effective_batch ≈ batch_size × (4 to 9)

# Keep:
# - ≤ 32 → safe
# - ≤ 64 → moderate
# - ≥ 128 → high risk (OOM)