#!/bin/bash -l
#SBATCH --job-name=guide
# #SBATCH -p longrun
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

# CLI
# echo "SPADE OFF"
# echo "Running best hyperparameters: seed 42"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 100 \
#     --batch_size 4 \
#     --patience 20 \
#     --num_workers 4 \
#     --seed 42 \
#     --lambda_nw 0.5779 \
#     --lambda_w 1.8142 \
#     --lambda_g 0.3537 \
#     --lambda_adversarial 0.0928 \
#     --lambda_perceptual 0.3942 \
#     --d_lr_scale 1.0 \
#     --use_spade false \
#     --run_name "baseline_best82" \
#     --group "cosia_aux2" \
#     --project TIR_SISR_v2 \


# echo "Running best hyperparameters: seed 0"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 100 \
#     --batch_size 4 \
#     --patience 20 \
#     --num_workers 4 \
#     --seed 0 \
#     --lambda_nw 0.5779 \
#     --lambda_w 1.8142 \
#     --lambda_g 0.3537 \
#     --lambda_adversarial 0.0928 \
#     --lambda_perceptual 0.3942 \
#     --d_lr_scale 1.0 \
#     --use_spade false \
#     --run_name "baseline_best82" \
#     --group "cosia_aux2" \
#     --project TIR_SISR_v2 \

# echo "Running with seed 123"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 100 \
#     --batch_size 4 \
#     --patience 20 \
#     --num_workers 4 \
#     --seed 123 \
#     --lambda_nw 0.9838 \
#     --lambda_w 2.2861 \
#     --lambda_g 0.2577 \
#     --lambda_adversarial 0.0911 \
#     --lambda_perceptual 0.4688 \
#     --d_lr_scale 1.0 \
#     --use_spade false \
#     --run_name "baseline_best84" \
#     --group "cosia_aux2" \
#     --project TIR_SISR_v2 \


echo "SPADE ON"
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 100 \
    --batch_size 4 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.6549 \
    --lambda_w 1.8146 \
    --lambda_g 0.1323 \
    --lambda_adversarial 0.0831 \
    --lambda_perceptual 0.6169 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "new_spade" \
    --group "cosia_aux3" \
    --project TIR_SISR_v2 \

# lambda_adversarial:
# 0.08306001528097819
# lambda_g:
# 0.13228874142501829
# lambda_nw:
# 0.6549198696398122
# lambda_perceptual:
# 0.6169104308821689
# lambda_w:
# 1.814636658431381


    # --lambda_nw 0.9233 \
    # --lambda_w 2.0522 \
    # --lambda_g 0.3018 \
    # --lambda_adversarial 0.0333 \
    # --lambda_perceptual 0.6288 \