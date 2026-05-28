#!/bin/bash -l
#SBATCH --job-name=sisr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
# #SBATCH --output=logs/01_fusion_aux_%j.log
# #SBATCH --output=logs/dir_aux_%j.log

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p logs checkpoints/dev_v4

python -m scripts.training.train \
    --model swinir \
    --lr 1e-4 \
    --batch_size 2 \
    --max_epochs 100 \
    --patience 30 \
    --num_workers 4 \
    --use_aux 1 \
    --lambda_grad 0.1 \
    --lambda_water 0.1 \
    --freeze_backbone 0 \
    --freeze_mode none \
    --project TIR_sisr \
    --run_name exp3 \
    --group spade_aux

# GAN training
# python -m scripts.training.train_gan \
#     --lr 1e-4 \
#     --batch_size 2 \
#     --max_epochs 100 \
#     --patience 30 \
#     --num_workers 4 \
#     --use_aux 1 \
#     --lambda_grad 0.0 \
#     --lambda_water 0.0 \
#     --water_weight 1.0 \
#     --adaptation_strategy projection \
#     --input_init pretrained_mean \
#     --freeze_backbone 0 \
#     --run_name 02_proj_full \
#     --group GAN

# # direct + pretrained mean — frozen  
# python -m scripts.training.train_realesrgan_aux \
#     --adaptation_strategy direct \
#     --input_init pretrained_mean \
#     --freeze_backbone 1 \
#     --run_name realesrgan_direct_mean_aux_frozen

# input_init options:pretrained_mean,gaussian,xavier,he, partial_preserve

# # no freezing
# --freeze_backbone 0

# # full backbone frozen
# --freeze_backbone 1 \ 
# --freeze_mode body

# # partial freeze
# --freeze_backbone 1 \
# --freeze_mode body+first