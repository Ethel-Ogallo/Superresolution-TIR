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
echo "Aux at input tuned:"
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 150 \
    --batch_size 4 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.9109 \
    --lambda_w 2.3314 \
    --lambda_g 0.1386 \
    --lambda_adversarial 0.0262 \
    --lambda_perceptual 0.4506 \
    --d_lr_scale 1.0 \
    --use_spade false \
    --run_name "aux" \
    --group "final" \
    --project TIR_sisr_final \


echo "SPADE:"
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 150 \
    --batch_size 4 \
    --patience 20 \
    --num_workers 4 \
    --lambda_nw 0.9109 \
    --lambda_w 2.3314 \
    --lambda_g 0.1386 \
    --lambda_adversarial 0.0262 \
    --lambda_perceptual 0.4506 \
    --d_lr_scale 1.0 \
    --use_spade true \
    --run_name "spade" \
    --group "final" \
    --project TIR_sisr_final 



# # lambda_adversarial:
# # 0.02617122129541192
# # lambda_g:
# # 0.1385885906493034
# # lambda_nw:
# # 0.910852112430372
# # lambda_perceptual:
# # 0.4505757489433283
# # lambda_w:
# # 2.331364297316022



