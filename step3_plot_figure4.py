import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import gpp_common as common

def plot_figure4_lines(df_binary, df_multi):
    """Replicate Figure 4: AUROC Learning Curves.
    
    Plots both P.1 (Binary) and P.2 (Multiclass).
    P.1 = Solid Lines
    P.2 = Dotted Lines
    """
    if df_binary.empty and df_multi.empty: return

    # Add Task Label
    if not df_binary.empty: df_binary['Task'] = 'P.1 (Binary)'
    if not df_multi.empty: df_multi['Task'] = 'P.2 (Multiclass)'
    
    # Combine
    df_all = pd.concat([df_binary, df_multi])
    df_all = df_all.reset_index(drop=True)
    
    fig, axes = plt.subplots(1, 3, figsize=(18, 6), sharey=True)
    models = ['M1', 'M2', 'M3']
    
    # Define style
    sns.set_style("whitegrid")
    
    # Custom Palette
    palette = {
        'GPP-Beta': '#1f77b4',       # Blue
        'GPP-Dirichlet': '#9467bd',  # Purple
        'LP': '#2ca02c',             # Green
        'SVM': '#ff7f0e',            # Orange
        'LPE': '#d62728',            # Red
        'GPP': '#9467bd'             # Fallback
    }
    
    # Line Styles
    # P.1 (Binary) -> Solid ("") or ((1,0))? Seaborn style mapping can be tricky.
    # Easier to use 'style' mapped to Task.
    
    for i, model in enumerate(models):
        ax = axes[i]
        
        data = df_all[df_all['model'] == model]
        if not data.empty:
            sns.lineplot(
                data=data, 
                x='n_obs', 
                y='AUROC', 
                hue='method', 
                style='Task', # Differentiate Solid vs Dotted
                style_order=['P.1 (Binary)', 'P.2 (Multiclass)'], # Ensure consistent order
                # Note: Seaborn defaults: P.1=Solid, P.2=Dashed/Dotted usually.
                palette=palette,
                markers=True,
                ax=ax,
                linewidth=2,
                err_style='band',
                errorbar=('ci', 95),
                legend=True if i==0 else False
            )
            
        ax.set_title(f'Model {model}', fontsize=14)
        ax.set_xlabel('Observations')
        if i == 0:
            ax.set_ylabel('AUROC')
        else:
            ax.set_ylabel('')
        
        ax.set_ylim(0.5, 1.02)
        ax.set_xscale('log')
        ax.set_xticks([2, 8, 32, 128])
        ax.set_xticklabels([2, 8, 32, 128])

    # Legend Adjustment
    # We need to ensure the legend shows both Colors (Method) and Styles (Task)
    if axes[0].get_legend():
        axes[0].legend(loc='lower right', bbox_to_anchor=(1.0, 0.05), fontsize='small', framealpha=0.9)

    # Save
    output_dir = os.path.join(common.FIGURE_DIR, "results_combined")
    os.makedirs(output_dir, exist_ok=True)
    
    plt.suptitle(f'Figure 4: Learning Curves (Solid=P.1, Dotted=P.2)', fontsize=16)
    plt.tight_layout()
    save_path = os.path.join(output_dir, f'figure4_auroc_combined.png')
    plt.savefig(save_path, dpi=300)
    print(f"Saved {save_path}")

def main():
    print("=== Step 3: Generating Figure 4 ===", flush=True)
    common.ensure_dirs()
    
    df_bin = pd.DataFrame()
    df_multi = pd.DataFrame()
    
    try:
        df_bin = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_auroc_binary.csv'))
    except: print("Warning: Binary results not found.")
    
    try:
        df_multi = pd.read_csv(os.path.join(common.CSV_DIR, 'results_jax_auroc_multi.csv'))
    except: print("Warning: Multiclass results not found.")

    plot_figure4_lines(df_bin, df_multi)

if __name__ == "__main__":
    main()

