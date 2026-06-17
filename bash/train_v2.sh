# #!/bin/bash -l
# #SBATCH --job-name=cosia
# # #SBATCH -p longrun
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=6
# #SBATCH --mem=32G 
# # #SBATCH --time=12:00:00 

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
# echo "Running best hyperparameters:"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 150 \
#     --batch_size 4 \
#     --patience 50 \
#     --num_workers 4 \
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

# echo "Running with 2nd best hyperparameters:"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 150 \
#     --batch_size 4 \
#     --patience 50 \
#     --num_workers 4 \
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

# echo "SPADE ON"
# echo "Running best hyperparameters:"
# python -m scripts.training.train \
#     --lr 1e-4 \
#     --max_epochs 150 \
#     --batch_size 4 \
#     --patience 50 \
#     --num_workers 4 \
#     --lambda_nw 0.5779 \
#     --lambda_w 1.8142 \
#     --lambda_g 0.3537 \
#     --lambda_adversarial 0.0928 \
#     --lambda_perceptual 0.3942 \
#     --d_lr_scale 1.0 \
#     --use_spade true \
#     --run_name "exp3_best82" \
#     --group "cosia_aux2" \
#     --project TIR_SISR_v2 \

echo "Running with 2nd best hyperparameters:"
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 150 \
    --batch_size 4 \
    --patience 50 \
    --num_workers 4 \
    --lambda_nw 0.9838 \
    --lambda_w 2.2861 \
    --lambda_g 0.2577 \
    --lambda_adversarial 0.0911 \
    --lambda_perceptual 0.4688 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "exp3_best84" \
    --group "cosia_aux2" \
    --project TIR_SISR_v2 \

# best lambdas
# lambda_adversarial:
# 0.09111855035655785
# lambda_g:
# 0.2576985417548614
# lambda_nw:
# 0.983774191477874
# lambda_perceptual:
# 0.4687885084292344
# lambda_w:
# 2.286066968410901
