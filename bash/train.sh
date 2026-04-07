#!/bin/bash -l
#SBATCH --job-name=sisr_edsr            # Job name
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-gpu=4                   # 4 CPU cores
#SBATCH --mem=16G                             # Memory per node

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p results/logs/edsr

# -------- Run training --------
python -m scripts.training.train  \
    --model edsr \
    --metadata_json data/full_metadata.json \
    --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth \
    --lr 1e-4 \
    --lambda_grad 0.5 \
    --bb_lr_scale 0.1 \
    --max_epochs 100 \
    --patience 30 \
    --batch_size 2 \
    --num_workers 2 \
    --freeze_backbone \
    --run_test \
    --project TIR_sisr \
    --run_name EDSR_frozen_v4 \
    --group 3c-repeat+data-aug
