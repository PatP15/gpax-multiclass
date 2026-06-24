#!/bin/bash
#SBATCH -c 8
#SBATCH -t 2:00:00
#SBATCH --mem=32G
#SBATCH -p sapphire
#SBATCH -o logs/%x_%j.out
#SBATCH -e logs/%x_%j.err

export HF_HOME=/n/netscratch/walsh_lab_seas/Everyone/ppuma/.cache/huggingface
export PYTHONNOUSERSITE=1
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH
export JAX_PLATFORMS=cpu

# Add project root to PYTHONPATH
export PYTHONPATH=$(pwd):$PYTHONPATH

module load python/3.10.12-fasrc01 

source ~/.bashrc
conda activate gpax-multiclass

echo "Running Step 3: Visualization on CPU..."
python -u annoMI/step3_visualize.py
