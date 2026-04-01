#!/bin/bash -l
#SBATCH --job-name=sisr_edsr_eval_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# Environment 
export WANDB_API_KEY="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"
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# Project root 
cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p results/logs/slurm

# Run evaluation 
# Checkpoint: wandb.ai → TIR_sisr → your run → Artifacts tab → copy path
# Format: ogalloethel-university-of-south-brittany/TIR_sisr/model-<run_id>:best
python -m scripts.evaluation.evaluate \
    --checkpoint   "wandb:ogalloethel-university-of-south-brittany/TIR_sisr/model-ejg57dm5:best" \
    --metadata_json data/full_metadata.json \
    --split        test \
    --num_workers  2 \
    --project      TIR_sisr \
    --run_name     EDSR_eval_test \
    --group        tir-3chn-repeat