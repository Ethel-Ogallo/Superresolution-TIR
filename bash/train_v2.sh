#!/bin/bash -l
#SBATCH --job-name=cosia
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
# mkdir -p logs checkpoints/gan_v2

# # CLI
# echo "Spade off with sobel spatial gradient:"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 80 \
#     --batch_size 4 \
#     --patience 15 \
#     --num_workers 4 \
#     --lambda_nw 0.353 \
#     --lambda_w 1.562 \
#     --lambda_g 1.381 \
#     --lambda_adversarial 0.061 \
#     --lambda_perceptual 0.791 \
#     --d_lr_scale 1.0 \
#     --use_spade false \
#     --run_name "exp6.1_loss" \
#     --group "spade" \
#     --project TIR_SISR_v2 \

# echo "Spade on with sobel spatial gradient:"
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 80 \
    --batch_size 4 \
    --patience 15 \
    --num_workers 4 \
    --lambda_nw 0.353 \
    --lambda_w 1.562 \
    --lambda_g 1.381 \
    --lambda_adversarial 0.061 \
    --lambda_perceptual 0.791 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "exp1" \
    --group "cosia_aux" \
    --project TIR_SISR_v2 \
