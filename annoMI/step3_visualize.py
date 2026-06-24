import os
import sys
import gc
import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datasets import load_dataset
import torch
import matplotlib
# Use Agg backend to avoid display errors on cluster
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# --- Monkeypatch for torch.utils._pytree compatibility with newer transformers ---
import torch.utils._pytree as pytree
def register_pytree_node_wrapper(cls, flatten_fn, unflatten_fn, serialized_type_name=None):
    return pytree._register_pytree_node(cls, flatten_fn, unflatten_fn)

if not hasattr(pytree, "register_pytree_node"):
    pytree.register_pytree_node = register_pytree_node_wrapper
# ---------------------------------------------------------------------------------

from transformers import AutoTokenizer, AutoModelForCausalLM
from tqdm import tqdm

# Add project root to path for GPax imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPax.probing import probabilistic_probe_multiclass as ppm
from GPax.probing import probabilistic_probe as pp

import annomi_common as common

def plot_auroc(df, filename, title):
    if df.empty:
        print(f"No data for {filename}")
        return
        
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='auroc', hue='Scenario', marker='o')
    plt.title(title)
    plt.ylabel("AUROC (Misclassification Detection)")
    plt.xlabel("Number of Training Samples")
    plt.ylim(0.4, 1.0)
    plt.legend(title='Method', loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    
    save_path = os.path.join(common.FIGURE_DIR, filename)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

def plot_figure6_lpe_vs_gpp(raw_lpe, raw_gpp):
    """
    Scatter plot of Aleatoric vs Epistemic Uncertainty.
    Compare LPE vs GPP-Dirichlet (Context).
    Hue = Sample Size (n).
    """
    print("Generating Figure 6 (LPE vs GPP)...")
    
    # Process Raw Data into DataFrame
    plot_data = []
    
    for label, raw_dict in [('LPE', raw_lpe), ('GPP-Dirichlet', raw_gpp)]:
        # raw_dict is NpzFile object, keys behave like list
        for key in raw_dict.files:
            if not key.endswith('_epistemic'): continue
            
            # Extract n from key "n10_epistemic"
            n_str = key.split('_')[0] # "n10"
            try:
                n = int(n_str[1:])
            except ValueError:
                continue
            
            epis = raw_dict[f'{n_str}_epistemic']
            alea = raw_dict[f'{n_str}_aleatoric']
            
            # Ensure 1D arrays
            if epis.ndim > 1: epis = epis.flatten()
            if alea.ndim > 1: alea = alea.flatten()
            
            # Sample if too many points to avoid clutter
            if len(epis) > 500:
                idx = np.random.choice(len(epis), 500, replace=False)
                epis = epis[idx]
                alea = alea[idx]
            
            for e, a in zip(epis, alea):
                plot_data.append({
                    'Method': label,
                    'n': n,
                    'Epistemic': e,
                    'Aleatoric': a
                })
                
    df = pd.DataFrame(plot_data)
    if df.empty:
        print("No raw data found for Figure 6.")
        return

    # Plot Normal
    g = sns.FacetGrid(df, col="Method", hue="n", palette="viridis", height=5, aspect=1.1)
    g.map(sns.scatterplot, "Epistemic", "Aleatoric", alpha=0.7, s=30)
    g.add_legend(title="Training Samples (n)")
    
    # Adjust axes labels and title
    g.set_axis_labels("Epistemic Uncertainty", "Aleatoric Uncertainty")
    g.set_titles(col_template="{col_name}")
    
    plt.subplots_adjust(top=0.85)
    g.fig.suptitle("Uncertainty Manifolds: LPE vs GPP-Dirichlet (Context)", fontsize=16)
    
    save_path = os.path.join(common.FIGURE_DIR, 'figure6_lpe_vs_gpp.png')
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved {save_path}")

    # Plot Zoomed (Smaller Y-axis, removing outliers)
    # Calculate 99th percentile for Y-axis limit
    y_max = df['Aleatoric'].quantile(0.99)
    x_max = df['Epistemic'].quantile(0.99)

    g2 = sns.FacetGrid(df, col="Method", hue="n", palette="viridis", height=5, aspect=1.1)
    g2.map(sns.scatterplot, "Epistemic", "Aleatoric", alpha=0.7, s=30)
    g2.add_legend(title="Training Samples (n)")
    
    g2.set_axis_labels("Epistemic Uncertainty", "Aleatoric Uncertainty")
    g2.set_titles(col_template="{col_name}")
    g2.set(ylim=(0, y_max * 1.1), xlim=(0, x_max * 1.1)) # Add 10% buffer
    
    plt.subplots_adjust(top=0.85)
    g2.fig.suptitle("Uncertainty Manifolds (Zoomed): LPE vs GPP-Dirichlet", fontsize=16)
    
    save_path_zoomed = os.path.join(common.FIGURE_DIR, 'figure6_lpe_vs_gpp_zoomed.png')
    plt.savefig(save_path_zoomed, dpi=300)
    plt.close()
    print(f"Saved {save_path_zoomed}")

def generate_examples_table(all_raw_data, sample_size=100):
    print(f"Generating Examples Table (n={sample_size})...")
    
    # 1. Load Original Data to get Text
    try:
        _, df_test = common.load_annomi_data()
    except Exception as e:
        print(f"Error loading original data: {e}")
        return

    # Verify alignment
    y_full = df_test['y_mot'].values
    
    # 3. Select 4 Interesting Examples
    
    # We need predictions from Context model (Standard GPP)
    if 'Context' not in all_raw_data:
        print("Context raw data missing for examples table.")
        return
        
    ctx_data = all_raw_data['Context']
    key_prefix = f"n{sample_size}_"
    
    if f'{key_prefix}y_pred' not in ctx_data:
        print(f"Sample size {sample_size} not found in Context raw data.")
        return
        
    preds = ctx_data[f'{key_prefix}y_pred']
    probs = ctx_data[f'{key_prefix}probs']
    y_true = ctx_data[f'{key_prefix}y_true']
    
    # Verify alignment
    if len(preds) != len(df_test):
        print(f"Warning: Test set size mismatch! DF: {len(df_test)}, Raw: {len(preds)}")
        return

    # Find candidate indices (relative to test set 0..N)
    correct_mask = (preds == y_true)
    incorrect_mask = ~correct_mask
    high_conf_mask = (np.max(probs, axis=1) > 0.8)
    low_conf_mask = (np.max(probs, axis=1) < 0.5)
    
    candidates = {}
    
    # High Conf Correct
    opts = np.where(correct_mask & high_conf_mask)[0]
    if len(opts) > 0: candidates['High Conf Correct'] = opts[0]
    
    # Low Conf Correct
    opts = np.where(correct_mask & low_conf_mask)[0]
    if len(opts) > 0: candidates['Low Conf Correct'] = opts[0]
    
    # High Conf Incorrect
    opts = np.where(incorrect_mask & high_conf_mask)[0]
    if len(opts) > 0: candidates['High Conf Incorrect'] = opts[0]
    
    # Disagreement: Context Wrong, Prompted Correct (if available)
    if 'Context (Expert)' in all_raw_data:
        p_data = all_raw_data['Context (Expert)']
        if f'{key_prefix}y_pred' in p_data:
            p_preds = p_data[f'{key_prefix}y_pred']
            p_correct = (p_preds == y_true)
            # Find where Context Wrong & Prompted Correct
            opts = np.where(incorrect_mask & p_correct)[0]
            if len(opts) > 0: candidates['Expert Fixes Context'] = opts[0]
        
    # Fill remaining slots if any missing
    used_indices = set(candidates.values())
    all_indices = np.arange(len(preds))
    np.random.shuffle(all_indices)
    
    needed = 4 - len(candidates)
    for i in all_indices:
        if needed <= 0: break
        if i not in used_indices:
            candidates[f'Random {needed}'] = i
            needed -= 1
            
    # Build Table
    records = []
    
    label_map = {0: 'Sustain', 1: 'Neutral', 2: 'Change'}
    
    for desc, idx in candidates.items():
        row_data = df_test.iloc[idx]
        
        record = {
            'Example Type': desc,
            'Therapist Text': row_data['therapist_text'],
            'Client Text': row_data['client_text'],
            'GT Label': label_map[row_data['y_mot']],
        }
        
        # Add model data
        for model_name, raw_d in all_raw_data.items():
            if f'{key_prefix}probs' not in raw_d: continue
            
            m_probs = raw_d[f'{key_prefix}probs'][idx]
            m_pred = raw_d[f'{key_prefix}y_pred'][idx]
            m_epis = raw_d[f'{key_prefix}epistemic'][idx]
            m_alea = raw_d[f'{key_prefix}aleatoric'][idx]
            
            # Ensure scalars
            if isinstance(m_epis, np.ndarray): m_epis = m_epis.item()
            if isinstance(m_alea, np.ndarray): m_alea = m_alea.item()
            if isinstance(m_pred, np.ndarray): m_pred = m_pred.item()
            
            record[f'{model_name} Pred'] = label_map[m_pred]
            record[f'{model_name} Probs'] = f"[{m_probs[0]:.2f}, {m_probs[1]:.2f}, {m_probs[2]:.2f}]"
            record[f'{model_name} Unc'] = f"{m_epis:.3f} / {m_alea:.3f}"
            
        records.append(record)
        
    df_table = pd.DataFrame(records)
    
    # Save
    csv_path = os.path.join(common.FIGURE_DIR, 'examples_table.csv')
    df_table.to_csv(csv_path, index=False)
    print(f"Saved examples table to {csv_path}")
    
    df_table.T.to_csv(os.path.join(common.FIGURE_DIR, 'examples_table_transposed.csv'))

def main():
    print("=== Step 3: Visualization ===")
    
    # 1. Load CSV Results
    dfs = []
    # Updated Names for Legend: (Prompted) -> (Expert)
    files = [
        (os.path.join(common.RESULTS_DIR, 'results_exp2.csv'), 'Context'),
        (os.path.join(common.RESULTS_DIR, 'results_exp4_quality.csv'), 'Context + Quality'),
        (os.path.join(common.RESULTS_DIR, 'results_exp2_prompted.csv'), 'Context (Expert)'),
        (os.path.join(common.RESULTS_DIR, 'results_exp4_prompted_quality.csv'), 'Context + Quality (Expert)'),
        (os.path.join(common.RESULTS_DIR, 'results_exp2_fewshot.csv'), 'Context (Expert Few-Shot)'),
        (os.path.join(common.RESULTS_DIR, 'results_lpe.csv'), 'LPE (Expert Few-Shot)')
    ]
    
    for fpath, name in files:
        if os.path.exists(fpath):
            try:
                df = pd.read_csv(fpath)
                df['Scenario'] = name
                dfs.append(df)
            except Exception as e:
                print(f"Error loading {fpath}: {e}")
                
    if not dfs:
        print("No results found.")
        return
        
    all_res = pd.concat(dfs, ignore_index=True)
    
    # 2. Plot Accuracy (All)
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_res, x='n', y='accuracy', hue='Scenario', marker='o')
    plt.title("Motivation Classification Accuracy vs Sample Size")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Training Samples")
    plt.legend(title='Method', loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(os.path.join(common.FIGURE_DIR, 'accuracy_comparison.png'), dpi=300)
    plt.close()
    print(f"Saved accuracy_comparison.png")
    
    # 3. Plot AUROC (All)
    plot_auroc(all_res, 'auroc_comparison.png', "Misclassification Detection (AUROC)")

    # Load Raw Data
    all_raw_data = {}
    
    # Map friendly name -> filename
    raw_files = {
        'Context': 'raw_exp2.npz',
        'Context + Quality': 'raw_exp4_quality.npz',
        'Context (Expert)': 'raw_exp2_prompted.npz',
        'Context + Quality (Expert)': 'raw_exp4_prompted_quality.npz',
        'Context (Expert Few-Shot)': 'raw_exp2_fewshot.npz',
        'LPE (Expert Few-Shot)': 'raw_lpe.npz'
    }
    
    for name, fname in raw_files.items():
        fpath = os.path.join(common.DATA_DIR, fname)
        if os.path.exists(fpath):
            try:
                all_raw_data[name] = np.load(fpath)
            except Exception as e:
                print(f"Error loading {fname}: {e}")
    
    # 5. Plot Figure 6 (LPE vs GPP Raw Data)
    if 'LPE (Expert Few-Shot)' in all_raw_data and 'Context' in all_raw_data:
        plot_figure6_lpe_vs_gpp(all_raw_data['LPE (Expert Few-Shot)'], all_raw_data['Context'])

    # 6. Generate Examples Table
    if all_raw_data:
        max_n = common.SAMPLE_SIZES[-1]
        generate_examples_table(all_raw_data, sample_size=max_n)

if __name__ == "__main__":
    main()
