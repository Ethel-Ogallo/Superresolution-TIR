#!/bin/bash -l
#SBATCH --job-name=wt_all
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G
#SBATCH --time=48:00:00  

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/gan_final

# -------- Step A: Initialize the Sweep Server-Side --------
# run this first
# wandb sweep configs/sweep.yaml
# This command will output a unique SWEEP_ID, which you need for the next step.
SWEEP_ID="3265lr6l"  # Replace with your actual SWEEP_ID from the previous command

# -------- Step B: Run the Agent --------
# NEW FIXES: Force Python to recognize your current directory folder
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# -------- Step B: Run the Agent --------
# Fixed the project path to match the exact string W&B registered
# --count 6 guarantees the script stops after 6 intelligent iterations.
wandb agent ogalloethel-university-of-south-brittany/TIR_SISR_final/$SWEEP_ID --count 30