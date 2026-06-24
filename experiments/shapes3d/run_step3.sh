#!/bin/bash
#SBATCH -p sapphire
#SBATCH -t 0:30:00
#SBATCH --mem=16G
#SBATCH -o logs/step3_%j.out
#SBATCH -e logs/step3_%j.err

# Load modules
module load python/3.10.12-fasrc01

# Activate conda
source /n/sw/Miniforge3-24.11.3-0/etc/profile.d/conda.sh || source ~/.bashrc
conda activate gpax-multiclass

# Plotting runs on CPU to avoid potential GPU resource issues on test partition
export JAX_PLATFORMS=cpu

echo "Generating Figure 4..."
python experiments/shapes3d/step3_plot_figure4.py

echo "Generating Figure 5..."
python experiments/shapes3d/step3_plot_figure5.py

echo "Generating Figure 6..."
python experiments/shapes3d/step3_plot_figure6.py

echo "Generating Manifold..."
python experiments/shapes3d/step3_plot_manifold.py
