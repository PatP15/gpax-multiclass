#!/bin/bash
#SBATCH -p gpu
#SBATCH -t 0:30:00
#SBATCH --mem=40G
#SBATCH --gres=gpu:1
#SBATCH -o logs/verification_%j.out
#SBATCH -e logs/verification_%j.err

# Load modules
module load python/3.10.12-fasrc01
module load cuda/12.2.0-fasrc01 cudnn/8.9.2.26_cuda12-fasrc01

# Activate conda environment
# Explicitly source conda.sh to ensure conda command is available in subshell
source ~/miniforge3/etc/profile.d/conda.sh || source ~/anaconda3/etc/profile.d/conda.sh || source ~/.bashrc
conda activate gpax-multiclass

# JAX Memory allocation flags to prevent OOM/Segfaults
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export XLA_PYTHON_CLIENT_ALLOCATOR=platform

# Run the verification script
echo "Starting verification run with 1k subset..."
JAX_PLATFORM_NAME=gpu python gpp_extended_verification.py
echo "Verification run completed."

