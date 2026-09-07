#!/bin/bash -l
#SBATCH --job-name=debug_env
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=2
#SBATCH --mem=4G

set -euxo pipefail

echo "START"

cd /share/castor/home/e2406751/Superresolution-TIR
echo "CD OK"

ls

source /share/common/anaconda/etc/profile.d/conda.sh
conda activate sisr

echo "CONDA OK"

which python
python -c "print('PYTHON OK')"