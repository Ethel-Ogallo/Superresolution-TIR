#!/bin/bash -l
#SBATCH --job-name=seq_sr
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp


source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/vsr

# -------- CLI --------
python -m scripts.training.seq_train \
    --lr 1e-4 \
    --max_epochs 150 \
    --batch_size 4 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.8123 \
    --lambda_w 1.7126 \
    --lambda_g 0.3207 \
    --run_name "exp3" \
    --group "baseline" \
    --project Sequential_TIR \