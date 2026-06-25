import numpy as np
import pandas as pd
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
import scipy.stats as stats
import matplotlib.pyplot as plt
import seaborn as sns
import sys
import os
import h5py

# Add repo root (dir containing GPax/) to sys.path, robust to cwd/location.
def _find_repo_root(start):
    d = os.path.dirname(os.path.abspath(start))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, 'GPax')):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(start))

_REPO_ROOT = _find_repo_root(__file__)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from GPax.probing import gp, gp_multiclass
from GPax.probing import probabilistic_probe as pp
import ontology
import cnn_model

def run_uncertainty_analysis():
    print("Starting Uncertainty Analysis (Figures 5 & 6)...")
    
    try:
        f = h5py.File(os.path.join(_REPO_ROOT, '3dshapes.h5'), 'r')
        N_SUBSET = 2000
        indices = np.random.choice(480000, N_SUBSET, replace=False)
        indices.sort()
        images = f['images'][indices].astype(np.float32) / 255.0
        labels_raw = f['labels'][indices]
        floor_hues = labels_raw[:, 0] # Index 0 is floor hue
        f.close()
    except Exception as e:
        print(f"Data load failed: {e}")
        return

    # We need to train a model first to get embeddings.
    # Train M3 (Color model)
    y_m3 = ontology.get_concept_labels(labels_raw)['M3']
    # Split
    X_train, X_test, y_train, y_test = train_test_split(images, y_m3, test_size=0.5, random_state=42)
    
    state = cnn_model.train_model(
        X_train, y_train, X_test, y_test, 
        num_classes=8, num_epochs=2, batch_size=32, verbose=False
    )
    embeddings = cnn_model.get_embeddings(state, images)
    
    # Define task: Floor Hue > 0.5
    y_true_binary = (floor_hues >= 0.5).astype(int)
    
    # Define "Soft" Ground Truth Probabilities
    # Sigmoid centered at 0.5 with slope 20
    gt_probs = 1 / (1 + np.exp(-20 * (floor_hues - 0.5)))
    
    # Split into Train/Test
    # Split embeddings first
    X_train, X_test, idx_train, idx_test = train_test_split(
        embeddings, np.arange(len(embeddings)), test_size=0.5, random_state=42
    )
    y_train = y_true_binary[idx_train]
    y_test = y_true_binary[idx_test]
    p_test = gt_probs[idx_test]
    
    judged_probs_gpp = []
    judged_probs_lpe = []
    
    # Run GPP Beta
    try:
        unc_gpp = pp.gpp(x_query=X_test, x_observed=X_train, y_observed=y_train)
        judged_probs_gpp = np.array(unc_gpp['Judged probability']).flatten()
        
        # Run LPE
        unc_lpe = pp.lpe(x_query=X_test, x_observed=X_train, y_observed=y_train)
        judged_probs_lpe = np.array(unc_lpe['Judged probability']).flatten()
        
        # --- Figure 5 Left: Correlation ---
        corr_gpp = np.corrcoef(p_test, judged_probs_gpp)[0, 1]
        corr_lpe = np.corrcoef(p_test, judged_probs_lpe)[0, 1]
        
        plt.figure(figsize=(6, 5))
        plt.scatter(p_test, judged_probs_gpp, alpha=0.5, label=f'GPP (r={corr_gpp:.2f})')
        plt.scatter(p_test, judged_probs_lpe, alpha=0.5, label=f'LPE (r={corr_lpe:.2f})')
        plt.plot([0, 1], [0, 1], 'k--')
        plt.xlabel('Ground Truth Probability')
        plt.ylabel('Judged Probability')
        plt.title('Calibration / Correlation (Fig 5 Left)')
        plt.legend()
        plt.savefig(os.path.join(_REPO_ROOT, 'fig5_correlation.png'))
        print("Saved fig5_correlation.png")
    except Exception as e:
        print(f"Correlation analysis failed: {e}")
        import traceback
        traceback.print_exc()
    
    # --- Figure 5 Middle/Right & Figure 6 ---
    # Pick a specific query point x* where GT Prob is ~0.5 (Ambiguous)
    if p_test.size > 0:
        idx_ambiguous = np.argmin(np.abs(p_test - 0.5))
        x_amb = X_test[idx_ambiguous:idx_ambiguous+1]
    else:
        print("No test data for uncertainty analysis.")
        return
    
    obs_counts = [2, 4, 8, 16, 32, 64, 128]
    
    res_n = []
    
    for n in obs_counts:
        # Subsample training data
        if n > len(X_train): break
        idx_sub = np.random.choice(len(X_train), n, replace=False)
        x_sub = X_train[idx_sub]
        y_sub = y_train[idx_sub]
        
        # GPP
        try:
            u_gpp = pp.gpp(x_query=x_amb, x_observed=x_sub, y_observed=y_sub)
            gpp_judged = u_gpp['Judged probability'].item()
            
            # Handle key variations
            gpp_epis = u_gpp.get('Episteme', u_gpp.get('episteme', np.nan))
            if hasattr(gpp_epis, 'item'): gpp_epis = gpp_epis.item()
            elif isinstance(gpp_epis, (np.ndarray, jax.Array)): gpp_epis = gpp_epis.flatten()[0]
            
            gpp_alea = u_gpp.get('Alea', u_gpp.get('alea', np.nan))
            if hasattr(gpp_alea, 'item'): gpp_alea = gpp_alea.item()
            elif isinstance(gpp_alea, (np.ndarray, jax.Array)): gpp_alea = gpp_alea.flatten()[0]
            
            res_n.append({
                'N': n, 'Method': 'GPP', 
                'Judged': gpp_judged, 'Episteme': gpp_epis, 'Alea': gpp_alea
            })
        except Exception as e:
            print(f"GPP failed for N={n}: {e}")
        
        # LPE
        if len(np.unique(y_sub)) < 2:
            continue
        else:
            try:
                u_lpe = pp.lpe(x_query=x_amb, x_observed=x_sub, y_observed=y_sub)
                lpe_judged = u_lpe['Judged probability'].item()
                
                lpe_epis = u_lpe.get('Episteme', u_lpe.get('information_gain', np.nan))
                if hasattr(lpe_epis, 'item'): lpe_epis = lpe_epis.item()
                elif isinstance(lpe_epis, (np.ndarray, jax.Array)): lpe_epis = lpe_epis.flatten()[0]
                
                lpe_alea = u_lpe.get('Alea', u_lpe.get('expected_aleatory_entropy', np.nan))
                if hasattr(lpe_alea, 'item'): lpe_alea = lpe_alea.item()
                elif isinstance(lpe_alea, (np.ndarray, jax.Array)): lpe_alea = lpe_alea.flatten()[0]
                
                res_n.append({
                    'N': n, 'Method': 'LPE', 
                    'Judged': lpe_judged, 'Episteme': lpe_epis, 'Alea': lpe_alea
                })
            except Exception as e:
                print(f"LPE failed for N={n}: {e}")
        
    df_n = pd.DataFrame(res_n)
    
    if df_n.empty:
        print("No results for Figure 5/6.")
        return

    # Plot Fig 5 Middle/Right (Judged vs Episteme)
    plt.figure(figsize=(10, 5))
    
    # Filter for GPP/LPE
    plt.subplot(1, 2, 1)
    lpe_data = df_n[df_n['Method']=='LPE']
    if not lpe_data.empty:
        sns.scatterplot(data=lpe_data, x='Episteme', y='Judged', size='N', hue='N', legend=False)
        plt.title('LPE: Judged vs Episteme')
        plt.ylim(0, 1)
    
    plt.subplot(1, 2, 2)
    gpp_data = df_n[df_n['Method']=='GPP']
    if not gpp_data.empty:
        sns.scatterplot(data=gpp_data, x='Episteme', y='Judged', size='N', hue='N')
        plt.title('GPP: Judged vs Episteme')
        plt.ylim(0, 1)
    
    plt.tight_layout()
    plt.savefig('fig5_judged_episteme.png')
    print("Saved fig5_judged_episteme.png")
    
    # Plot Fig 6 (Alea vs Episteme evolution)
    plt.figure(figsize=(8, 6))
    sns.scatterplot(data=df_n, x='Episteme', y='Alea', hue='Method', style='Method', s=100)
    
    for i, row in df_n.iterrows():
        if not np.isnan(row['Episteme']) and not np.isnan(row['Alea']):
            plt.text(row['Episteme'], row['Alea'], str(int(row['N'])))
            
    plt.title('Alea vs Episteme Evolution (N=2..128) for Ambiguous Input')
    plt.savefig('fig6_alea_episteme.png')
    print("Saved fig6_alea_episteme.png")

if __name__ == "__main__":
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    run_uncertainty_analysis()
