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

echo "Running Step 2: Prompted Experiments (Context & Context+Quality & Few-Shot) on CPU..."
python -u annoMI/step2_exp2_prompted.py
python -u annoMI/step2_exp4_prompted_quality.py
python -u annoMI/step2_exp2_fewshot.py
