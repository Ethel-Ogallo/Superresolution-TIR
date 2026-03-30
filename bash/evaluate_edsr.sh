#!/bin/bash -l
#SBATCH --job-name=sisr_edsr_eval_test
#SBATCH --output=results/logs/eval_edsr_test_%j.out
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=01:00:00

# -------- Environment --------
export WANDB_API_KEY="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"   
source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Run evaluation --------
python -m scripts.evaluation.evaluate \
       --checkpoint wandb:entity/project/model-ID:best \
       --metadata_csv data/metadata.csv \
       --split test \
       --num_workers 2 \
       --save_outputs \
       --n_vis 5 \
       --project TIR_sisr \
       --run_name EDSR_eval_test