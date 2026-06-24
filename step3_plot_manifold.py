import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import gpp_common as common

def plot_dirichlet_manifold_3d(probs, title, filename, save_data_path=None):
    """Plot 3-class Dirichlet manifold in 3D and save data."""
    if probs.shape[1] != 3:
        print("Manifold plot requires 3 classes.")
        return

    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    
    p = ax.scatter(probs[:, 0], probs[:, 1], probs[:, 2], c=probs[:, 0], cmap='viridis', alpha=0.6, s=20)
    
    ax.set_xlabel('P(Class 0)')
    ax.set_ylabel('P(Class 1)')
    ax.set_zlabel('P(Class 2)')
    ax.set_title(title)
    
    fig.colorbar(p, ax=ax, label='P(Class 0)', shrink=0.5, aspect=10)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1); ax.set_zlim(0, 1)
    ax.view_init(elev=30, azim=45)
    
    plt.tight_layout()
    plt.savefig(filename, dpi=300)
    print(f"Saved {filename}")
    plt.close()

def plot_dirichlet_manifold(probs, title, filename):
    """Plot 3-class Dirichlet manifold (2D Projection)."""
    if probs.shape[1] != 3:
        print("Manifold plot requires 3 classes.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    vertices = np.array([v1, v2, v3])
    
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    probs_2d = probs @ vertices
    scatter = ax.scatter(probs_2d[:, 0], probs_2d[:, 1], c=probs[:, 0], cmap='viridis', alpha=0.6, s=30)
    
    ax.text(v1[0]-0.05, v1[1]-0.05, 'Class 0', ha='right')
    ax.text(v2[0]+0.05, v2[1]-0.05, 'Class 1', ha='left')
    ax.text(v3[0], v3[1]+0.05, 'Class 2', ha='center')
    
    ax.set_xlim(-0.2, 1.2); ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    ax.set_title(title)
    plt.colorbar(scatter, label='P(Class 0)')
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()

def main():
    print("=== Step 3: Generating Manifold Plots ===", flush=True)
    common.ensure_dirs()
    
    try:
        csv_path = os.path.join(common.MANIFOLD_DIR, "manifold_probs.csv")
        if not os.path.exists(csv_path):
            print("No manifold data found. Run step2_run_probes.py first.")
            return
            
        df = pd.read_csv(csv_path)
        probs = df[['P_Class0', 'P_Class1', 'P_Class2']].values
        
        # 2D Plot
        plot_dirichlet_manifold(probs, "GPP Dirichlet Manifold (3-Class Shape)", 
                               os.path.join(common.MANIFOLD_DIR, "figure_manifold_3class_2d.png"))
        
        # 3D Plot
        plot_dirichlet_manifold_3d(probs, "GPP Dirichlet Manifold 3D (3-Class Shape)", 
                                  os.path.join(common.MANIFOLD_DIR, "figure_manifold_3class_3d.png"))
                                  
    except Exception as e:
        print(f"Error plotting manifold: {e}")

if __name__ == "__main__":
    main()

