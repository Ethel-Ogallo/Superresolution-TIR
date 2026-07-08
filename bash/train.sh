#!/bin/bash -l
#SBATCH --job-name=bench
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:0  

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/phase2.1

# EDSR 
echo "Training EDSR model..."
python -m scripts.training.train \
    --model edsr \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --project TIR_sisr_final* \
    --run_name edsr\
    --group benchmark

# SWINIR
echo "Training SWINIR model..."
python -m scripts.training.train \
    --model swinir \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --project TIR_sisr_final* \
    --run_name swinir\
    --group benchmark

# HAT
echo "Training HAT model..."
python -m scripts.training.train \
    --model hat \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --project TIR_sisr_final* \
    --run_name hat\
    --group benchmark

# REALESGRAN
echo "Training REALESGRAN model..."
python -m scripts.training.train \
    --model realesrgan \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 20 \
    --num_workers 4 \
    --project TIR_sisr_final* \
    --run_name realesrgan\
    --group benchmark
