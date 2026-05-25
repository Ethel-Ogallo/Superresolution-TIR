#!/bin/bash -l
#SBATCH --job-name=sisr_eval_phase1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=32G
# #SBATCH --output=logs/eval_dev_%j.log
# #SBATCH --error=logs/eval_phase1_%j.err

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p logs results/phase1

# Run all models
# python -m scripts.evaluation.eval \
#     --all \
#     --phase 1

# phase 1
# python -m scripts.evaluation.eval \
#     --model resshift \
#     --phase 1 \
#     --use_water_metrics \
#     --run_name "03_resshift_p1" \
#     --group benchmark_phase1

# phase 2
# python -m scripts.evaluation.eval \
#     --model swinir \
#     --phase 2 \
#     --use_water_metrics \
#     --project TIR_sisr \
#     --run_name 01_proj_aux_eval \
#     --group model_dev \
#     --checkpoint checkpoints/model_dev/phase1/swinir/model_dev-epoch=34-val_full_psnr=34.4880.ckpt

python -m scripts.evaluation.eval \
  --model swinir \
  --checkpoint checkpoints/dev_v2/swinir/best-epoch=26-val_full_psnr=17.7527-v1.ckpt \
  --use_aux 1 \
  --project TIR_sisr \
  --run_name 05_dir_eval \
  --group model_dev_v2 \