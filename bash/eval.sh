#!/bin/bash -l
#SBATCH --job-name=sisr_edsr_eval_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=6
#SBATCH --mem=24G

# Environment 
export WANDB_API_KEY="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"
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# Project root 
cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p results/logs/slurm

# Run evaluation 
python -m scripts.evaluation.evaluate \
    --model       edsr \
    --checkpoint   "wandb:ogalloethel-university-of-south-brittany/TIR_sisr/model-mwqw8evc:best" \
    --metadata_json data/full_metadata.json \
    --split        test \
    --output_dir   results/SR_images/ \
    --num_workers  2 \
    --project      TIR_sisr \
    --run_name     eval_full_PDR \
    --group        campaign_splits
