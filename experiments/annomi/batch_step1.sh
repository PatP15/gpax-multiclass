#!/bin/bash
#SBATCH -c 8
#SBATCH -t 6:00:00
#SBATCH --mem=256G
#SBATCH -p seas_gpu
#SBATCH --gres=gpu:nvidia_a100-sxm4-80gb:1
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

export HF_HOME=/n/netscratch/walsh_lab_seas/Everyone/ppuma/.cache/huggingface
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export JAX_PLATFORMS=cpu
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

# Load modules (if needed, but prefer Conda)
module load python/3.10.12-fasrc01 
module load cuda/12.4.1-fasrc01

# Source Conda
source ~/.bashrc
conda activate gpax-multiclass

# Run Step 1 Unprompted (Context, Context+Qual)
echo "Running Step 1 (Process Data Unprompted)..."
python -u experiments/annomi/step1_process_data.py
