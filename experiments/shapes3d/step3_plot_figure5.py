import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy.stats import pearsonr
import gpp_common as common

def plot_figure5_correlation(df_fuzz, scenario_name):
    """Replicate Figure 5 (Revised): 
    
    Panel 1: Pearson Correlation vs Observations.
    Panel 2-4: Judged Probability vs Episteme for LPE, GPP-Beta, GPP-Dirichlet (at GT=0.5).
    """
    if df_fuzz.empty: return
    
    fig, axes = plt.subplots(1, 4, figsize=(22, 5))
    
    # --- Panel 1: Pearson Correlation vs Observations ---
    ax1 = axes[0]
    
    # Calculate Pearson R per method per n_obs
    # We need to aggregate across GT levels? 
    # The Pearson correlation is between "GT Prob" and "Judged Prob" ACROSS all queries.
    # Since we have data for GT=[0.25, 0.5, 0.75, 1.0] in the dataframe, we can compute correlation using all these points.
    
    pearson_results = []
    methods = ['LPE', 'GPP-Beta', 'GPP-Dirichlet']
    obs_levels = sorted(df_fuzz['n_obs'].unique())
    
    for m in methods:
        for n in obs_levels:
            # Filter data for this method and n_obs
            d = df_fuzz[(df_fuzz['method'] == m) & (df_fuzz['n_obs'] == n)]
            if len(d) > 10:
                r, _ = pearsonr(d['gt_prob'], d['judged_prob'])
                pearson_results.append({'method': m, 'n_obs': n, 'pearson': r})
                
    df_pearson = pd.DataFrame(pearson_results)
    
    palette = {
        'GPP-Beta': '#1f77b4',       # Blue
        'GPP-Dirichlet': '#9467bd',  # Purple
        'LPE': '#d62728',            # Red
    }
    
    sns.lineplot(data=df_pearson, x='n_obs', y='pearson', hue='method', palette=palette, marker='o', ax=ax1)
    ax1.set_title('Pearson Correlation vs Obs')
    ax1.set_xlabel('Observations')
    ax1.set_ylabel('Pearson Coeff')
    ax1.set_xscale('log')
    ax1.set_xticks([2, 8, 32, 128])
    ax1.set_xticklabels([2, 8, 32, 128])
    ax1.set_ylim(0, 1.05)
    ax1.legend(title='Method', loc='lower right')

    # --- Panels 2, 3, 4: Judged Prob vs Episteme (GT=0.5) ---
    target_gt = 0.5
    
    # For global legend extraction
    obs_handles = []
    obs_labels = []
    
    for i, method in enumerate(methods):
        ax = axes[i+1]
        
        # Filter for method and GT ≈ 0.5
        d = df_fuzz[(df_fuzz['method'] == method) & (df_fuzz['gt_prob'].between(target_gt-0.05, target_gt+0.05))]
        
        if not d.empty:
            # Scatter plot: X=Episteme, Y=Judged Prob, Hue=n_obs
            # Downsample if too many points
            if len(d) > 2000: d = d.sample(2000, random_state=42)
            
            # Force n_obs to categorical for discrete legend
            d = d.copy()
            # d['n_obs'] = d['n_obs'].astype(str) # Optional, but ensures discrete
            
            sns.scatterplot(
                data=d, 
                x='episteme', 
                y='judged_prob', 
                hue='n_obs', 
                palette='rocket_r', 
                ax=ax,
                s=10,
                alpha=0.7,
                legend='full' if i==2 else False
            )
            
            if i == 2: # Last plot
                if ax.get_legend():
                    obs_handles, obs_labels = ax.get_legend_handles_labels()
                    ax.get_legend().remove()
        
        ax.set_title(f'{method} (GT={target_gt})')
        ax.set_xlabel('Episteme')
        ax.set_xscale('log')
        if i == 0:
            ax.set_ylabel('Judged Probability')
        else:
            ax.set_ylabel('')
            
        ax.set_ylim(-0.05, 1.05)

    # Global Legend for Observations
    if obs_handles:
        fig.legend(obs_handles, obs_labels, title='Observations', loc='center right')

    plt.tight_layout()
    plt.subplots_adjust(right=0.92) # Make space for legend
    
    # Save
    output_dir = os.path.join(common.FIGURE_DIR, f"results_{scenario_name}")
    os.makedirs(output_dir, exist_ok=True)
    save_path = os.path.join(output_dir, f'figure5_combined_{scenario_name}.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

def main():
    print("=== Step 3: Generating Figure 5 (Fuzziness) ===", flush=True)
    common.ensure_dirs()
    
    # P.1 Binary
    try:
        df_p1 = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p1.csv'))
        plot_figure5_correlation(df_p1, "Binary_P1")
    except Exception as e:
        print(f"Could not plot Fig 5 P1: {e}")
        
    # P.2 Multiclass
    try:
        df_p2 = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p2.csv'))
        plot_figure5_correlation(df_p2, "Multiclass_P2")
    except Exception as e:
        print(f"Could not plot Fig 5 P2: {e}")

if __name__ == "__main__":
    main()

