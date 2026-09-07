#!/bin/bash -l
#SBATCH --job-name=aux_sisr
#SBATCH -p longrun
#SBATCH --gres=gpu:1
#SBATCH -w sn1
#SBATCH --cpus-per-task=6
#SBATCH --mem=32G

# -------- Environment --------
export WANDB_API_KEY=""   
export WANDB_DIR=/tmp

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

# -------- Project root (all relative paths resolve from here) --------
cd /share/castor/home/e2406751/Superresolution-TIR
# mkdir -p logs checkpoints/dev_v3

# echo "projection input with aux channels"
python -m scripts.training.train_gan \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 30 \
    --num_workers 4 \
    --use_aux 1 \
    --adaptation_strategy projection \
    --freeze_backbone 0\
    --project TIR_sisr_final \
    --run_name projected4 \
    --group aux_ablation

echo "direct aux input with pretrained mean"
python -m scripts.training.train_gan \
    --lr 1e-4 \
    --batch_size 4 \
    --max_epochs 100 \
    --patience 30 \
    --num_workers 4 \
    --use_aux 1 \
    --adaptation_strategy direct \
    --input_init pretrained_mean \
    --freeze_backbone 0\
    --project TIR_sisr_final \
    --run_name direct4 \
    --group aux_ablation

# # direct + pretrained mean — frozen  
# python -m scripts.training.train_realesrgan_aux \
#     --adaptation_strategy direct \
#     --input_init pretrained_mean \
#     --freeze_backbone 1 \
#     --run_name realesrgan_direct_mean_aux_frozen

# input_init options:pretrained_mean,gaussian,xavier,he, partial_preserve

# # no freezing
# --freeze_backbone 0

# # full backbone frozen
# --freeze_backbone 1 \ 
# --freeze_mode body

# # partial freeze
# --freeze_backbone 1 \
# --freeze_mode body+first