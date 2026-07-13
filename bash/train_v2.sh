#!/bin/bash -l
#SBATCH --job-name=spade
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G 
# #SBATCH --time=12:00:00 

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/home/e2406751/Superresolution-TIR
# mkdir -p logs checkpoints/gan_v6

# CLI
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 150 \
    --batch_size 4 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.6030 \
    --lambda_w 1.7601 \
    --lambda_g 0.2598 \
    --lambda_adversarial 0.0708 \
    --lambda_perceptual 0.3642 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "exp2" \
    --group "spade" \
    --project TIR_sisr_final \

# lambda_adversarial:
# 0.0707898130215005
# lambda_g:
# 0.25982136202473927
# lambda_nw:
# 0.603015214572084
# lambda_perceptual:
# 0.3642004331550006
# lambda_w:
# 1.7600751516568551
