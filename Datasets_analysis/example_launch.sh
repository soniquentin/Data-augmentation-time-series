#!/bin/bash -l
#SBATCH -J MyGPUJob
#SBATCH --mail-type=end,fail
#SBATCH --mail-user=quentin.lao@polytechnique.edu
#SBATCH -N 1
#SBATCH -n 2
#SBATCH -c 7
#SBATCH --gpus=2
#SBATCH --time=47:59:00
#SBATCH -p gpu


conda activate tf-gpu
TF_CPP_MIN_LOG_LEVEL=2 python make_tests.py 0
conda deactivate
