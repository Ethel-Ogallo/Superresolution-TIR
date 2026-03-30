#!/bin/bash
#SBATCH --job-name=sisr_edsr_train          # Job name
#SBATCH --output=results/logs/train_edsr_%j.out  # stdout log (%j = job ID)
#SBATCH --gres=gpu:1                   # Request 1 GPU
#SBATCH --cpus-per-task=4              # Adjust if needed
#SBATCH --mem=32G                       # Memory per node
#SBATCH --time=24:00:00                # Max runtime (HH:MM:SS)

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Run training --------
python -m scripts.train.train_edsr \
       --metadata_csv data/metadata.csv \
       --pretrained data/pretrained/edsr_baseline_x4.pth \
       --n_feats 64 \
       --n_blocks 16 \
       --lr 1e-4 \
       --bb_lr_scale 0.1 \
       --max_epochs 20 \
       --batch_size 2 \
       --num_workers 4 \
       --project TIR_sisr \
       --run_name EDSR_frozen_nf64_lr1e-4 \
       #--unfreeze        # Uncomment if fine-tuning full backbone