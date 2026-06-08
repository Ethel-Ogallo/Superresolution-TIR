# #!/bin/bash -l
# #SBATCH --job-name=opt
# #SBATCH --gres=gpu:1
# #SBATCH --cpus-per-task=6
# #SBATCH --mem=32G 
# #SBATCH --time=12:00:00 

# -------- Environment --------
export WANDB_API_KEY="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"   # W&B API key for non-interactive login
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root --------
cd /share/home/e2406751/Superresolution-TIR
mkdir -p logs checkpoints/gan_v3

# final run of best
echo "Starting final training run with best hyperparameters..."
python -m scripts.training.train \
    --lr 1e-4 \
    --max_epochs 1 \
    --batch_size 4 \
    --patience 0 \
    --num_workers 4 \
    --lambda_nw 0.30927002874069115 \
    --lambda_w 3.277213095769832 \
    --lambda_g 0.5123973224335376 \
    --lambda_adversarial 0.004983185326619845 \
    --lambda_perceptual 0.09591394888871198 \
    --use_spade False \
    --run_name "test" \
    --group "spade" \
    --project TIR_SISR_v2 
