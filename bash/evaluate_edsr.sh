#!/bin/bash
#SBATCH --job-name=sisr_edsr_eval          # Job name
#SBATCH --output=results/logs/eval_edsr_%j.out  # stdout log (%j = job ID)
#SBATCH --gres=gpu:1                        # Request 1 GPU
#SBATCH --cpus-per-task=4                   # Adjust if needed
#SBATCH --mem=16G                           # Memory per node
#SBATCH --time=04:00:00                     # Max runtime (HH:MM:SS)

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Run evaluation --------
python -m scripts.evaluation.evaluate \
       --checkpoint wandb:entity/project/model-ID:best \
       --metadata_csv data/metadata.csv \
       --split test \
       --num_workers 4 \
       --save_outputs \
       --n_vis 5 \
       --project TIR_sisr \
       --run_name EDSR_eval_test