#!/bin/bash -l
#SBATCH --job-name=train
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
    --lambda_nw 0.8123 \
    --lambda_w 1.7126 \
    --lambda_g 0.3207 \
    --lambda_adversarial 0.0247 \
    --lambda_perceptual 0.7037 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "exp4_sans_aug" \
    --group "spade" \
    --project TIR_SISR_v3 \

# lambda_adversarial:
# 0.024693810190593295
# lambda_g:
# 0.3207001143579563
# lambda_nw:
# 0.8122826678726971
# lambda_perceptual:
# 0.7036859701800203
# lambda_w:
# 1.712567198414096
