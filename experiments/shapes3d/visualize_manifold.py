import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import dirichlet
import jax
import jax.numpy as jnp
import pandas as pd

def plot_simplex_density(alphas, label):
    """
    Plot Dirichlet density on a 2D simplex (triangle).
    For K=3, alpha = [a1, a2, a3].
    We project 3D barycentric coordinates to 2D.
    """
    # Define 2D projection of 3 corners
    corners = np.array([[0, 0], [1, 0], [0.5, np.sqrt(3)/2]])
    
    # Generate grid on simplex
    res = 100
    x = np.linspace(0, 1, res)
    y = np.linspace(0, 1, res)
    XX, YY = np.meshgrid(x, y)
    
    # Convert 2D grid to 3D barycentric coords (p1, p2, p3)
    # p1 + p2 + p3 = 1
    # x = p2 + 0.5 * p3
    # y = 0.866 * p3
    # -> p3 = y / 0.866
    # -> p2 = x - 0.5 * p3
    # -> p1 = 1 - p2 - p3
    
    p3 = YY / (np.sqrt(3)/2)
    p2 = XX - 0.5 * p3
    p1 = 1 - p2 - p3
    
    # Filter valid points inside simplex
    mask = (p1 >= 0) & (p2 >= 0) & (p3 >= 0)
    
    # Compute PDF
    pdf = np.zeros_like(XX)
    
    # Scipy Dirichlet requires input (N, 3)
    pts = np.vstack([p1[mask], p2[mask], p3[mask]]).T
    # Add epsilon to avoid boundary zeros if alpha < 1
    pts = np.clip(pts, 1e-5, 1.0)
    pts = pts / pts.sum(axis=1, keepdims=True)
    
    try:
        densities = dirichlet.pdf(pts.T, alphas)
        pdf[mask] = densities
    except Exception as e:
        print(f"PDF error: {e}")
        
    # Plot
    plt.figure(figsize=(6, 5))
    plt.contourf(XX, YY, pdf, levels=20, cmap='viridis')
    plt.colorbar(label='Density')
    
    # Draw triangle boundary
    plt.plot([0, 1, 0.5, 0], [0, 0, np.sqrt(3)/2, 0], 'k-', lw=2)
    plt.text(0, -0.1, 'Class 1', ha='center')
    plt.text(1, -0.1, 'Class 2', ha='center')
    plt.text(0.5, np.sqrt(3)/2 + 0.05, 'Class 3', ha='center')
    plt.title(f'Dirichlet Density: {label}\nAlpha={alphas}')
    plt.axis('equal')
    plt.axis('off')
    plt.tight_layout()
    plt.savefig(f'manifold_{label}.png')
    print(f"Saved manifold_{label}.png")

def run_3class_visualization():
    print("Generating Dirichlet Manifold Visualizations...")
    
    # Scenario 1: Confident prediction for Class 1 (Corner)
    # High alpha for class 1, low for others
    alpha_corner = [10.0, 1.0, 1.0]
    plot_simplex_density(alpha_corner, 'Confident_Class1')
    
    # Scenario 2: High Epistemic Uncertainty (Center mass)
    # Low alphas everywhere (e.g. prior)
    alpha_uncertain = [1.1, 1.1, 1.1] # Flat-ish but centered
    plot_simplex_density(alpha_uncertain, 'Uncertain_Prior')
    
    # Scenario 3: High Aleatoric Uncertainty (Center mass but peaked)
    # High alphas everywhere -> confident that it's a mix (ambiguous input)
    alpha_ambiguous = [10.0, 10.0, 10.0]
    plot_simplex_density(alpha_ambiguous, 'Ambiguous_HighCount')
    
if __name__ == "__main__":
    run_3class_visualization()

