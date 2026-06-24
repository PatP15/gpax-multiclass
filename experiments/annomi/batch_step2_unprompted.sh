#!/bin/bash
#SBATCH -c 32
#SBATCH -t 12:00:00
#SBATCH --mem=128G
#SBATCH -p sapphire
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

export HF_HOME=/n/netscratch/walsh_lab_seas/Everyone/ppuma/.cache/huggingface
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

module load python/3.10.12-fasrc01 

source ~/.bashrc
conda activate gpax-multiclass

echo "Running Step 2: Unprompted Experiments (Context & Context+Quality) on CPU..."
python -u experiments/annomi/step2_exp2_context.py
python -u experiments/annomi/step2_exp4_quality.py
