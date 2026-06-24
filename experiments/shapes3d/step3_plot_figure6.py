import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gpp_common as common

def plot_figure6_uncertainty(df_fuzz, scenario_name):
    """Replicate Figure 6 (Revised): Uncertainty Analysis on Fuzzy Data.
    
    4-Panel Layout:
    1. LPE (GT=0.5)
    2. GPP-Beta (GT=0.5)
    3. GPP-Dirichlet (GT=0.5)
    4. GPP-Dirichlet (GT=1.0)
    """
    if df_fuzz.empty: return
    
    # Config
    panels = [
        ('LPE', 0.5),
        ('GPP-Beta', 0.5),
        ('GPP-Dirichlet', 0.5),
        ('GPP-Dirichlet', 1.0)
    ]
    
    fig, axes = plt.subplots(2, 2, figsize=(12, 10), sharex=True, sharey=True)
    axes = axes.flatten()
    
    for i, (method, gt) in enumerate(panels):
        ax = axes[i]
        
        # Filter
        # Use small range for float comparison
        d = df_fuzz[(df_fuzz['method'] == method) & (df_fuzz['gt_prob'].between(gt-0.05, gt+0.05))]
        
        if not d.empty:
            if len(d) > 2000: d = d.sample(2000, random_state=42)
            sns.scatterplot(
                data=d, 
                x='episteme', 
                y='alea', 
                hue='n_obs', 
                palette='rocket_r', 
                ax=ax,
                legend=(i==0) # Legend only on first
            )
        else:
            ax.text(0.5, 0.5, "No Data", ha='center', transform=ax.transAxes)
            
        ax.set_title(f'{method} | GT ≈ {gt}')
        ax.set_xlabel('Epistemic Uncertainty')
        ax.set_ylabel('Aleatoric Uncertainty')
        ax.set_xscale('log')
        ax.set_yscale('log')
        
    plt.suptitle(f'Uncertainty Decomposition ({scenario_name})', fontsize=16)
    plt.tight_layout()
    
    # Save
    output_dir = os.path.join(common.FIGURE_DIR, f"results_{scenario_name}")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'figure6_uncertainty_{scenario_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

def main():
    print("=== Step 3: Generating Figure 6 (Fuzziness) ===", flush=True)
    common.ensure_dirs()
    
    # P.1 Binary
    try:
        df_p1 = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p1.csv'))
        plot_figure6_uncertainty(df_p1, "Binary_P1")
    except Exception as e:
        print(f"Could not plot Fig 6 P1: {e}")
        
    # P.2 Multiclass
    try:
        df_p2 = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p2.csv'))
        plot_figure6_uncertainty(df_p2, "Multiclass_P2")
    except Exception as e:
        print(f"Could not plot Fig 6 P2: {e}")

if __name__ == "__main__":
    main()
