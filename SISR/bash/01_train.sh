#!/bin/bash -l
#SBATCH --job-name=bench
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH -w sn1
#SBATCH --time=48:00:0  

# -------- Environment --------
export WANDB_API_KEY=""   
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/castor/home/e2406751/Superresolution-TIR

# EDSR 
echo "Training EDSR model..."
python -m scripts.training.train \
    --model edsr \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 100 \
    --num_workers 4 \
    --project TIR_sisr_final \
    --run_name edsr\
    --group benchmark_final

# SWINIR
echo "Training SWINIR model..."
python -m scripts.training.train \
    --model swinir \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 100 \
    --num_workers 4 \
    --project TIR_sisr_final \
    --run_name swinir\
    --group benchmark_final

# HAT
echo "Training HAT model..."
python -m scripts.training.train \
    --model hat \
    --lr 1e-4 \
    --batch_size 2 \
    --max_epochs 100 \
    --patience 100 \
    --num_workers 4 \
    --project TIR_sisr_final \
    --run_name hat\
    --group benchmark_final

# REALESGRAN
echo "Training REALESGRAN model..."
python -m scripts.training.train \
    --model realesrgan \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 100 \
    --num_workers 4 \
    --project TIR_sisr_final \
    --run_name realesrgan\
    --group benchmark_final
