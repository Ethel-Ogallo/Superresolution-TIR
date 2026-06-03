#!/bin/bash -l
#SBATCH --job-name=sisr
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=24:00:00
# #SBATCH --output=logs/exp5_%j.log

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/dev_gan


# ==============================================================================
# RUN 1.1: Global Baseline (Standard Super-Resolution)
# Objective: Equal weight to land and water. Shows how massive landscape 
# variations dominate the gradient and mask minor river thermal plumes.
# ==============================================================================
echo "Starting Run 1.1: Global Baseline..."

python -m scripts.training.train \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 1.0 \
    --lambda_w 1.0 \
    --lambda_g 0.0 \
    --lambda_perceptual 0.0 \
    --lambda_adversarial 0.0 \
    --project TIR_SISR \
    --group gan_ablation \
    --run_name exp1.1_baseline

# ==============================================================================
# RUN 1.2: River-Heavy Targeted Focus
# Objective: Heavily penalizes errors inside the river channel mask. Forces 
# the generator to recover subtle in-stream temperature variations.
# ==============================================================================
echo "Starting Run 1.2: River-Heavy Focus..."

python -m scripts.training.train \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.2 \
    --lambda_w 2.0 \
    --lambda_g 0.0 \
    --lambda_perceptual 0.0 \
    --lambda_adversarial 0.0 \
    --project TIR_SISR \
    --group gan_ablation \
    --run_name exp1.2_river_focus

