#!/bin/bash -l
#SBATCH --job-name=sisr_train_phase2
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
# #SBATCH --output=logs/train_phase2_%j.log
# #SBATCH --error=logs/train_phase2_%j.err

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/phase2

python -m scripts.training.train \
    --model resshift \
    --lr 1e-4 \
    --batch_size 1 \
    --max_epochs 100 \
    --patience 10 \
    --num_workers 4 \
    --project TIR_sisr \
    --run_name resshift_ph2 \
    --group benchmark_phase2

# SWINIR
# python -m scripts.training.train \
#     --model swinir \
#     --metadata_json data/full_metadata.json \
#     --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/SwinIR_classical_x4.pth \
#     --lr 1e-4 \
#     --lambda_grad 0.1 \
#     --bb_lr_scale 0.1 \
#     --max_epochs 100 \
#     --patience 30 \
#     --batch_size 2 \
#     --num_workers 2 \
#     --run_name exp4_fr_256 \
#     --group SWIN \
#     --project TIR_sisr \
#     --freeze_backbone

# HAT
# # NOTE: change patch size to 64 before running HAT
# python -m scripts.training.train \
#     --model hat \
#     --metadata_json data/full_metadata.json \
#     --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/HAT_imagenet_x4.pth \
#     --lr 1e-4 \
#     --lambda_grad 0.1 \
#     --bb_lr_scale 0.1 \
#     --max_epochs 100 \
#     --patience 30 \
#     --batch_size 2 \
#     --num_workers 2 \
#     --run_name exp3_fr_256 \
#     --group HAT \
#     --project TIR_sisr \
#     --freeze_backbone

# REALESGRAN
# python -m scripts.training.train \
#     --model real_esrgan \
#     --metadata_json data/full_metadata.json \
#     --pretrained /share/home/e2406751/Superresolution-TIR/data/pretrained/RealESRGAN_generator_x4.pth \
#     --pretrained_d /share/home/e2406751/Superresolution-TIR/data/pretrained/RealESRGAN_discriminator_x4.pth \
#     --lr 1e-4 \
#     --lambda_grad 0.1 \
#     --bb_lr_scale 0.1 \
#     --max_epochs 100 \
#     --patience 30 \
#     --batch_size 2 \
#     --num_workers 2 \
#     --run_name exp3_fr_256 \
#     --group Real-ESRGAN \
#     --project TIR_sisr \
#     --freeze_backbone


# EDSR: /share/home/e2406751/Superresolution-TIR/data/pretrained/EDSR_baseline_x4.pth
# SwinIR: /share/home/e2406751/Superresolution-TIR/data/pretrained/SwinIR_classical_x4.pth
# HAT: /share/home/e2406751/Superresolution-TIR/data/pretrained/HAT_imagenet_x4.pth
# realESRGAN: /share/home/e2406751/Superresolution-TIR/data/pretrained/RealESRGAN_discriminator_x4.pth
# realESRGAN: /share/home/e2406751/Superresolution-TIR/data/pretrained/RealESRGAN_generator_x4.pth

