#!/bin/bash -l
#SBATCH --job-name=sisr_edsrv1            # Job name
#SBATCH --error=results/logs/slurm/train_edsr_%j.err
#SBATCH --gres=gpu:1                        # 1 GPU
#SBATCH --cpus-per-gpu=4                   # 4 CPU cores
#SBATCH --mem=8G                             # Memory per node

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
    --metadata_json data/full_metadata.json \
    --pretrained   /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth \
    --n_feats      64 \
    --n_blocks     16 \
    # --unfreeze \  #enable to fine-tune all layers 
    --lr           1e-4 \
    --bb_lr_scale  0.1 \
    --max_epochs   30 \
    --batch_size   1 \
    --num_workers  2 \
    --run_test \
    --project      TIR_sisr \
    --run_name     EDSR_frozen_v1 \
    --group        tir-3chn-repeat 
