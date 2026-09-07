#!/bin/bash -l
#SBATCH --job-name=seq_sr_aux
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -w sn5
#SBATCH --time=168:00:00 

# -------- Environment --------
export WANDB_API_KEY=""   
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/home/e2406751/Superresolution-TIR
# mkdir -p logs checkpoints/vsr

# -------- CLI --------
python -m scripts.training.seq_train \
    --lr 1e-4 \
    --max_epochs 80 \
    --batch_size 16 \
    --accumulate_grad_batches 2 \
    --patience 20 \
    --num_workers 6 \
    --lambda_nw 0.9109 \
    --lambda_w 2.3314 \
    --lambda_g 0.1386 \
    --use_aux \
    --n_aux_channels 25 \
    --run_name "aux_exp4" \
    --group "auxiliary" \
    --project Sequential_TIR


# python -m scripts.training.seq_train \
#     --lr 1e-4 \
#     --max_epochs 80 \
#     --batch_size 16 \
#     --accumulate_grad_batches 4 \
#     --patience 20 \
#     --num_workers 6 \
#     --lambda_nw 1.0 \
#     --lambda_w 2.0 \
#     --lambda_g 0.1 \
#     --run_name "exp4" \
#     --group "baseline_overlap_5perc" \
#     --project Sequential_TIR

