#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 3:00:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/step2_%j.out
#SBATCH -e logs/step2_%j.err

# Load modules
module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01

# Activate conda environment
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh || source ~/.bashrc
conda activate gpax-multiclass

# JAX Memory Flags
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

echo "Running Step 2: Probing Experiments..."
JAX_PLATFORM_NAME=gpu python experiments/shapes3d/step2_run_probes.py

