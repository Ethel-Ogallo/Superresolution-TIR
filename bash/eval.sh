#!/bin/bash -l
#SBATCH --job-name=sisr
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G

# -------- Environment --------
export WANDB_API_KEY=""  
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p logs results/phase1

# Run all models
python -m scripts.evaluation.eval \
    --model realesrgan \
    --phase 1 \
    --use_water_metrics \
    --run_name "pt_realesrgan" \
    --group benchmark

# phase 2
# python -m scripts.evaluation.eval \
#     --model edsr \
#     --phase 2 \
#     --use_water_metrics \
#     --run_name "03_edsr_eval" \
#     --group benchmark_phase2 \
#     --checkpoint checkpoints/phase2/edsr/edsr_phase2_epoch=69_val_psnr=20.9760.ckpt

# checkpoints/phase2/edsr/edsr_phase2_epoch=69_val_psnr=20.9760.ckpt
# checkpoints/phase2/hat/hat_phase2_epoch=16_val_psnr=21.2140.ckpt
# checkpoints/phase2/realesrgan/realesrgan_phase2_epoch=33_val_psnr=19.9362.ckpt
# checkpoints/phase2/swinir/swinir_phase2_epoch=21_val_psnr=21.6563.ckpt