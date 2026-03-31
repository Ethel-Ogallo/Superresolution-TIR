#!/bin/bash -l
#SBATCH --job-name=sisr_edsr_test          # Job name
#SBATCH --output=results/logs/slurm/train_edsr_%j.out
#SBATCH --error=results/logs/slurm/train_edsr_%j.err
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-task=2                   # 2 CPU cores
#SBATCH --mem=8G                             # Memory per node
#SBATCH --time=02:00:00                      # Max runtime

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p results/logs/slurm

# -------- Run training --------
python -m scripts.training.train_edsr \
    --metadata_csv data/metadata.csv \
    --pretrained   /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth \
    --n_feats      64 \
    --n_blocks     16 \
    --lr           1e-4 \
    --bb_lr_scale  0.1 \
    --max_epochs   20 \
    --batch_size   1 \
    --num_workers  2 \
    --project      TIR_sisr \
    --run_name     EDSR_slurm_test \
    --group        tir-3chn-repeat 
