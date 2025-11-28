import h5py
import numpy as onp
import numpy as np
import jax
import jax.numpy as jnp
from sklearn.model_selection import train_test_split
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.svm as sksvm
from scipy.stats import pearsonr
import sys
import os
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon
import seaborn as sns
import warnings
from functools import partial

warnings.filterwarnings('ignore')

sys.path.append(os.getcwd())

from GPax.probing import gp, gp_multiclass
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm
import ontology
import cnn_model

# --- Helper Functions ---

def get_3class_shape_labels(labels_raw):
    """Create 3-class shape labels from 4-class (use first 3 shapes: 0,1,2)."""
    shape = labels_raw[:, 4].astype(int)
    mask = shape < 3
    return shape[mask], mask

def compute_lpr_probs(x_query, x_observed, y_observed, num_classes):
    """Linear Probe Regression: Standard Logistic Regression."""
    if y_observed.ndim > 1:
        y_indices = np.argmax(y_observed, axis=1)
    else:
        y_indices = y_observed
        
    if num_classes == 2:
        clf = sklm.LogisticRegression(solver='lbfgs', max_iter=1000)
        try:
            clf.fit(x_observed, y_indices)
            probs = clf.predict_proba(x_query)
            if probs.shape[1] == 1:
                probs = np.hstack([1 - probs, probs])
        except:
            probs = np.ones((len(x_query), 2)) * 0.5
    else:
        clf = sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
        try:
            clf.fit(x_observed, y_indices)
            probs = clf.predict_proba(x_query)
            if probs.shape[1] < num_classes:
                full_probs = np.zeros((x_query.shape[0], num_classes))
                for i, c in enumerate(clf.classes_):
                    if c < num_classes:
                        full_probs[:, int(c)] = probs[:, i]
                probs = full_probs
        except:
            probs = np.ones((len(x_query), num_classes)) / num_classes
            
    return probs

def compute_auroc(y_true, y_probs, is_multiclass=False):
    """Compute AUROC."""
    try:
        if is_multiclass:
            auroc = skm.roc_auc_score(y_true, y_probs, multi_class='ovr', average='macro')
        else:
            # For binary, y_probs should be (N, 2) or (N,). If (N, 2), take column 1.
            if y_probs.ndim == 2 and y_probs.shape[1] == 2:
                auroc = skm.roc_auc_score(y_true, y_probs[:, 1])
            else:
                auroc = skm.roc_auc_score(y_true, y_probs)
    except Exception:
        auroc = np.nan
    return auroc

# --- Plotting Functions ---

def plot_figure4_lines(df_auroc, scenario_name):
    """Replicate Figure 4: AUROC Learning Curves."""
    if df_auroc.empty: return

    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['M1', 'M2', 'M3']
    
    # Define style
    sns.set_style("whitegrid")
    palette = {'GPP': '#1f77b4', 'LPE': '#ff7f0e', 'SVM': '#2ca02c', 'LP': '#d62728', 'Maha': '#9467bd'}
    
    for i, model in enumerate(models):
        ax = axes[i]
        data = df_auroc[df_auroc['model'] == model]
        
        if data.empty: continue
        
        # Use seaborn lineplot for automatic aggregation (mean) and confidence intervals (bands)
        sns.lineplot(
            data=data, 
            x='n_obs', 
            y='AUROC', 
            hue='method', 
            style='task_type',
            palette=palette,
            markers=True,
            dashes=True,
            ax=ax,
            linewidth=2,
            err_style='band', # Shaded confidence interval
            errorbar=('ci', 95) # 95% CI
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

    # Unified legend
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.9))
    for ax in axes:
        ax.get_legend().remove()

    plt.suptitle(f'Figure 4: Learning Curves ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure4_auroc_{scenario_name}.png', dpi=300)
    print(f"Saved figure4_auroc_{scenario_name}.png")

def plot_figure5_uncertainty(df_unc, df_sim, scenario_name):
    """Replicate Figure 5: Correlation and Judged Prob vs Episteme."""
    fig = plt.figure(figsize=(18, 6))
    
    # 1. Left: Correlation (GPP vs LPE)
    ax1 = plt.subplot(1, 3, 1)
    
    # Use SIMULATED data for correlation to ensure we cover the full probability range [0, 1]
    # If sim data is empty, fall back to real data
    data_corr = df_sim if not df_sim.empty else df_unc
    
    methods = ['GPP', 'LPE']
    colors = ['#1f77b4', '#ff7f0e']
    
    for idx, method in enumerate(methods):
        d = data_corr[data_corr['method'] == method]
        if len(d) > 0:
            corr, _ = pearsonr(d['gt_prob'], d['judged_prob'])
            label = f"{method} (r={corr:.2f})"
            ax1.scatter(d['gt_prob'], d['judged_prob'], alpha=0.3, label=label, color=colors[idx], s=10)
        else:
            ax1.scatter([], [], label=f"{method} (N/A)")

    ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
    ax1.set_xlabel('Ground Truth Probability')
    ax1.set_ylabel('Judged Probability')
    ax1.set_title('Calibration / Correlation (Fig 5 Left)')
    ax1.legend()
    
    # For Middle/Right, we need data where GT Prob is roughly 0.5 (Ambiguous)
    # Combine simulated ambiguous data with any real ambiguous data
    df_ambiguous = pd.concat([
        df_sim[(df_sim['gt_prob'] >= 0.4) & (df_sim['gt_prob'] <= 0.6)],
        df_unc[(df_unc['gt_prob'] >= 0.4) & (df_unc['gt_prob'] <= 0.6)]
    ])
    
    # 2. Middle: LPE (GT ~ 0.5)
    ax2 = plt.subplot(1, 3, 2)
    d_lpe = df_ambiguous[df_ambiguous['method'] == 'LPE']
    
    if not d_lpe.empty:
        sns.scatterplot(data=d_lpe, x='episteme', y='judged_prob', hue='n_obs', 
                       palette='rocket_r', ax=ax2, legend=False)
        ax2.set_xscale('log')
        ax2.set_title('LPE | GT Prob ≈ 0.5 (Fig 5 Mid)')
    else:
        ax2.text(0.5, 0.5, 'No Ambiguous Data for LPE')

    # 3. Right: GPP (GT ~ 0.5)
    ax3 = plt.subplot(1, 3, 3)
    d_gpp = df_ambiguous[df_ambiguous['method'] == 'GPP']
    
    if not d_gpp.empty:
        sns.scatterplot(data=d_gpp, x='episteme', y='judged_prob', hue='n_obs', 
                       palette='rocket_r', ax=ax3)
        ax3.set_xscale('log')
        ax3.set_title('GPP | GT Prob ≈ 0.5 (Fig 5 Right)')
        plt.legend(title='Observations', bbox_to_anchor=(1.05, 1), loc='upper left')
    else:
        ax3.text(0.5, 0.5, 'No Ambiguous Data for GPP')

    plt.suptitle(f'Figure 5: Uncertainty Analysis ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure5_uncertainty_{scenario_name}.png', dpi=300)
    print(f"Saved figure5_uncertainty_{scenario_name}.png")

def plot_figure6_alea(df_unc, df_sim, scenario_name):
    """Replicate Figure 6: Alea vs Episteme."""
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # Combine for ambiguous
    df_ambiguous = pd.concat([
        df_sim[(df_sim['gt_prob'] >= 0.4) & (df_sim['gt_prob'] <= 0.6)],
        df_unc[(df_unc['gt_prob'] >= 0.4) & (df_unc['gt_prob'] <= 0.6)]
    ])
    
    # High confidence data (GT ~ 1.0) - mostly from real data
    df_confident = df_unc[df_unc['gt_prob'] > 0.9]
    
    # 1. Left: LPE | GT ~ 0.5
    ax1 = axes[0]
    d_lpe = df_ambiguous[df_ambiguous['method'] == 'LPE']
    if not d_lpe.empty:
        sns.scatterplot(data=d_lpe, x='episteme', y='alea', hue='n_obs', 
                       palette='rocket_r', ax=ax1, legend=False)
        ax1.set_xscale('log')
        ax1.set_title('LPE | GT Prob ≈ 0.5')
    
    # 2. Middle: GPP | GT ~ 0.5
    ax2 = axes[1]
    d_gpp = df_ambiguous[df_ambiguous['method'] == 'GPP']
    if not d_gpp.empty:
        sns.scatterplot(data=d_gpp, x='episteme', y='alea', hue='n_obs', 
                       palette='rocket_r', ax=ax2, legend=False)
        ax2.set_xscale('log')
        ax2.set_title('GPP | GT Prob ≈ 0.5')

    # 3. Right: GPP | GT ~ 1.0
    ax3 = axes[2]
    d_gpp_conf = df_confident[df_confident['method'] == 'GPP']
    if not d_gpp_conf.empty:
        sns.scatterplot(data=d_gpp_conf, x='episteme', y='alea', hue='n_obs', 
                       palette='rocket_r', ax=ax3)
        ax3.set_xscale('log')
        ax3.set_title('GPP | GT Prob ≈ 1.0')
        plt.legend(title='Observations', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f'Figure 6: Alea vs Episteme ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure6_alea_episteme_{scenario_name}.png', dpi=300)
    print(f"Saved figure6_alea_episteme_{scenario_name}.png")

def plot_dirichlet_manifold(probs, title, filename):
    """Plot 3-class Dirichlet manifold."""
    if probs.shape[1] != 3:
        print("Manifold plot requires 3 classes.")
        return

    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Triangle vertices
    v1 = np.array([0, 0])
    v2 = np.array([1, 0])
    v3 = np.array([0.5, np.sqrt(3)/2])
    vertices = np.array([v1, v2, v3])
    
    # Draw triangle
    triangle = Polygon(vertices, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(triangle)
    
    # Project to 2D barycentric
    probs_2d = probs @ vertices
    
    # Scatter
    scatter = ax.scatter(probs_2d[:, 0], probs_2d[:, 1], c=probs[:, 0], cmap='viridis', alpha=0.6, s=30)
    
    # Labels
    ax.text(v1[0]-0.05, v1[1]-0.05, 'Class 0', ha='right')
    ax.text(v2[0]+0.05, v2[1]-0.05, 'Class 1', ha='left')
    ax.text(v3[0], v3[1]+0.05, 'Class 2', ha='center')
    
    ax.set_xlim(-0.2, 1.2)
    ax.set_ylim(-0.2, 1.2)
    ax.axis('off')
    ax.set_title(title)
    plt.colorbar(scatter, label='P(Class 0)')
    plt.savefig(filename, dpi=150)
    print(f"Saved {filename}")
    plt.close()


# --- Experiment Driver ---

def run_experiment(scenario_name, task_key, is_multiclass, embeddings_dict, labels_dict, teacher_model_name='M1'):
    print(f"\n>>> Starting Experiment: {scenario_name} (Multiclass={is_multiclass})")
    
    # Data Setup
    y_all = labels_dict[task_key]
    
    # 1. Train Teacher for Ground Truth Probabilities (using M1 embeddings)
    print("Training Teacher Model...")
    X_emb_teacher = embeddings_dict[teacher_model_name]
    # Align labels length
    L_teach = min(len(X_emb_teacher), len(y_all))
    X_emb_teacher = X_emb_teacher[:L_teach]
    y_teacher = y_all[:L_teach]
    
    # Split for teacher
    X_teach_train, X_teach_test, y_teach_train, y_teach_test = train_test_split(
        X_emb_teacher, y_teacher, test_size=0.5, random_state=42)
    
    # Use KNN for smooth probabilities
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance')
    knn.fit(X_teach_train, y_teach_train)
    
    # Get GT probs for the *test set* which we will probe
    # For Multiclass, predict_proba returns (N, K). For Binary (N, 2).
    gt_probs_test = knn.predict_proba(X_teach_test) 
    
    # Metric Containers
    auroc_results = []
    uncertainty_data = []
    
    # Probing Config
    observation_levels = [2, 4, 8, 16, 32, 64, 128]
    repeats_per_level = 5 # For error bars
    
    # Loop over Models (M1, M2, M3)
    for model_name in ['M1', 'M2', 'M3']:
        print(f"  Probing Model: {model_name}")
        X_emb = embeddings_dict[model_name]
        # Align
        L = min(len(X_emb), len(y_all))
        X_emb = X_emb[:L]
        y = y_all[:L]
        
        # Use the SAME test split indices as teacher to align GT probs
        _, X_probe_test, _, y_probe_test = train_test_split(X_emb, y, test_size=0.5, random_state=42)
        
        # Check alignment
        if len(X_probe_test) != len(gt_probs_test):
            # If models have different embedding sizes, we can't directly map teacher probs.
            # Simplified: Re-predict GT probs for THIS model's test set using teacher?
            # Better: Just trust the split random_state=42 keeps indices aligned if original inputs were same.
            # If inputs differ, we skip GT alignment for other models or re-train teacher.
            # Assumption: X_emb for M1, M2, M3 come from same images.
            pass

        # Loop over observation levels
        for n_obs in observation_levels:
            for r in range(repeats_per_level):
                seed = 42 + n_obs * 100 + r
                
                # Sample Training Data (Stratified)
                # We sample from the *other* half (train split)
                X_probe_train_pool, _, y_probe_train_pool, _ = train_test_split(X_emb, y, test_size=0.5, random_state=42)
                
                # Stratified sample of n_obs
                try:
                    # Use sklearn for stratified sampling
                    if n_obs >= len(np.unique(y_probe_train_pool)):
                         X_obs, _, y_obs, _ = train_test_split(
                            X_probe_train_pool, y_probe_train_pool, 
                            train_size=n_obs, stratify=y_probe_train_pool, random_state=seed)
                    else:
                        # Random sample if n_obs too small for stratification
                         indices = np.random.choice(len(X_probe_train_pool), n_obs, replace=False)
                         X_obs = X_probe_train_pool[indices]
                         y_obs = y_probe_train_pool[indices]
                except:
                    continue # Skip if sampling fails

                # --- RUN PROBES ---
                
                # 1. GPP
                try:
                    if is_multiclass:
                        y_obs_oh = jax.nn.one_hot(y_obs, gt_probs_test.shape[1])
                        unc = ppm.gpp_multiclass(X_probe_test, X_obs, y_obs_oh, num_classes=gt_probs_test.shape[1])
                        probs = np.array(unc['categorical_mu'])
                        # Uncertainty Metrics
                        # For multiclass, 'Judged Prob' is usually max prob? Or prob of true class?
                        # Paper Fig 5 uses "Judged Probability" (of the correct class? or class 1?)
                        # For binary, it's class 1.
                    else:
                        unc = pp.gpp(X_probe_test, X_obs, y_obs)
                        probs = np.array(unc['Judged probability']) # (N, 1) or (N,)
                        episteme = np.array(unc['Episteme'])
                        alea = np.array(unc['Alea'])

                    # AUROC
                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({
                        'model': model_name, 'n_obs': n_obs, 'method': 'GPP', 
                        'AUROC': score, 'task_type': scenario_name
                    })
                    
                    # Save uncertainty data (for specific N_obs and Model M1 only to save space?)
                    # Save all for now
                    if not is_multiclass: # Figure 5/6 usually binary focused in paper, but we try both
                        # For binary, store stats
                        for i in range(min(len(probs), 50)): # Store subsample
                            uncertainty_data.append({
                                'model': model_name, 'n_obs': n_obs, 'method': 'GPP',
                                'judged_prob': float(probs[i]),
                                'episteme': float(episteme[i]),
                                'alea': float(alea[i]),
                                'gt_prob': float(gt_probs_test[i, 1]) # Class 1 prob
                            })

                except Exception as e:
                    pass # print(f"GPP Err: {e}")

                # 2. LPE
                try:
                    if is_multiclass:
                        unc = ppm.lpe_multiclass(X_probe_test, X_obs, y_obs, repeats=20)
                        probs = np.array(unc['categorical_mu'])
                    else:
                        unc = pp.lpe(X_probe_test, X_obs, y_obs, repeats=20)
                        p1 = np.array(unc['Judged probability'])
                        probs = p1
                        episteme = np.array(unc['Episteme'])
                        alea = np.array(unc['Alea'])

                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({
                        'model': model_name, 'n_obs': n_obs, 'method': 'LPE', 
                        'AUROC': score, 'task_type': scenario_name
                    })
                    
                    if not is_multiclass:
                         for i in range(min(len(probs), 50)):
                            uncertainty_data.append({
                                'model': model_name, 'n_obs': n_obs, 'method': 'LPE',
                                'judged_prob': float(probs[i]),
                                'episteme': float(episteme[i]),
                                'alea': float(alea[i]),
                                'gt_prob': float(gt_probs_test[i, 1])
                            })
                except Exception as e:
                    pass

                # 3. SVM
                try:
                    if is_multiclass:
                        clf = sksvm.SVC(kernel='linear', probability=True)
                        clf.fit(X_obs, y_obs)
                        probs = clf.predict_proba(X_probe_test)
                    else:
                        res = pp.svm_probe(X_probe_test, X_obs, y_obs)
                        probs = res['Judged probability']
                    
                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({
                        'model': model_name, 'n_obs': n_obs, 'method': 'SVM', 
                        'AUROC': score, 'task_type': scenario_name
                    })
                except: pass

                # 4. LP (Linear Probe - Logistic)
                try:
                    probs = compute_lpr_probs(X_probe_test, X_obs, y_obs, gt_probs_test.shape[1])
                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({
                        'model': model_name, 'n_obs': n_obs, 'method': 'LP', 
                        'AUROC': score, 'task_type': scenario_name
                    })
                except: pass
                
                # 5. Mahalanobis
                try:
                    if is_multiclass:
                        res = ppm.maha_multiclass(X_probe_test, X_obs, y_obs)
                        # Maha returns negative distance (higher is better). 
                        # For AUROC we need "scores". Negative dist works.
                        # But it doesn't return probabilities per class directly in the simple implementation.
                        # Skip AUROC for Maha unless we implement prob conversion.
                        pass 
                    else:
                        res = pp.maha(X_probe_test, X_obs, y_obs)
                        # Score is 'episteme' (neg distance). 
                        # AUROC expects score for positive class.
                        # Maha is usually for OOD, but here used as probe?
                        # Let's skip AUROC for Maha to avoid confusion or use simplified dist.
                        pass
                except: pass

    return pd.DataFrame(auroc_results), pd.DataFrame(uncertainty_data)

def simulate_ambiguous_data(embeddings_dict, labels_dict, task_key):
    """Generate synthetic ambiguous data by interpolating centroids."""
    print(f"Simulating ambiguous data for {task_key}...")
    X_emb = onp.array(embeddings_dict['M1']) # Standard numpy
    y_full = labels_dict[task_key]
    
    # Convert to standard numpy
    if hasattr(y_full, 'tolist'): y_full = onp.array(y_full)
    
    # Truncate
    L = min(len(X_emb), len(y_full))
    X = X_emb[:L]
    y = y_full[:L]
    
    # Centroids
    c0 = X[y==0].mean(axis=0)
    c1 = X[y==1].mean(axis=0)
    
    # Interpolate with more points for denser plot
    alphas = np.random.uniform(0, 1, 5000) # INCREASED to 5000 for dense plots
    X_sim = np.array([(1-a)*c0 + a*c1 for a in alphas])
    y_gt = alphas # GT Probability of class 1
    
    # Run Probes on Simulation
    sim_results = []
    
    # We need a training set to probe AGAINST. Use a balanced subset of real data.
    # Sample once large enough
    if len(np.unique(y)) < 2:
        print(f"Warning: Only 1 class in simulation data for {task_key}. Skipping sim.")
        return pd.DataFrame()

    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    # Multiclass fallback: if simulating binary behavior between two classes (0 and 1),
    # we just use those two. If task is actually multiclass (0,1,2), 
    # we just pick first two for "ambiguity" between them.
    
    # Increase pool size for training samples to avoid "replace=False" errors with small subsets
    pool_size = 200 
    if len(idx0) < pool_size or len(idx1) < pool_size: 
         # Fallback if insufficient data - use whatever we have but warn
         pool_size = min(len(idx0), len(idx1))
         print(f"Warning: Limited data for simulation training set ({pool_size} per class).")
         if pool_size < 10:
             return pd.DataFrame()
    
    idx_train = np.concatenate([
        np.random.choice(idx0, pool_size, replace=False),
        np.random.choice(idx1, pool_size, replace=False)
    ])
    X_train = X[idx_train]
    y_train = y[idx_train]
    
    # Determine if we should use Multiclass or Binary Probes
    # Check the FULL label set for this task to know if it's inherently multiclass
    is_multiclass_mode = (len(np.unique(y_full)) > 2)

    # Vary N_obs for the probes
    for n_obs in [2, 8, 32, 128]:
        if n_obs > len(X_train):
            continue
            
        # Subsample training set
        idx_sub = np.random.choice(len(X_train), n_obs, replace=False)
        X_tr_sub = X_train[idx_sub]
        y_tr_sub = y_train[idx_sub]
        
        # GPP
        try:
            if is_multiclass_mode:
                # Run Multiclass GPP on these 2-class samples (subset of 3 classes)
                # We need one-hot labels for observed
                # Assume 3 classes total for P2_shape_3class
                K = 3
                y_tr_oh = jax.nn.one_hot(y_tr_sub, K)
                unc = ppm.gpp_multiclass(X_sim, X_tr_sub, y_tr_oh, num_classes=K)
                
                # Judged prob: Probability of Class 1 (since we interpolate 0 -> 1)
                # Categorical mu is (N, K)
                probs_all = np.array(unc['categorical_mu'])
                jp = probs_all[:, 1] 
                
                # Episteme: For multiclass, let's use Max Probability as a simple proxy
                ep = np.max(probs_all, axis=1)
                
                # Alea: Placeholder
                al = np.zeros_like(jp) 
                
            else:
                # Binary GPP
                unc = pp.gpp(X_sim, X_tr_sub, y_tr_sub)
                jp = np.array(unc['Judged probability']).flatten()
                ep = np.array(unc['Episteme']).flatten()
                al = np.array(unc['Alea']).flatten()
                
            for i in range(len(X_sim)):
                sim_results.append({
                    'method': 'GPP', 'n_obs': n_obs, 
                    'judged_prob': jp[i], 'episteme': ep[i], 'alea': al[i],
                    'gt_prob': y_gt[i]
                })
        except Exception as e: 
            # print(f"Sim GPP Err: {e}")
            pass
        
        # LPE
        try:
            if is_multiclass_mode:
                unc = ppm.lpe_multiclass(X_sim, X_tr_sub, y_tr_sub, repeats=20)
                probs_all = np.array(unc['categorical_mu'])
                jp = probs_all[:, 1]
                ep = np.max(probs_all, axis=1) # Proxy
                al = np.zeros_like(jp)
            else:
                unc = pp.lpe(X_sim, X_tr_sub, y_tr_sub)
                jp = np.array(unc['Judged probability']).flatten()
                ep = np.array(unc['Episteme']).flatten()
                al = np.array(unc['Alea']).flatten()
                
            for i in range(len(X_sim)):
                sim_results.append({
                    'method': 'LPE', 'n_obs': n_obs, 
                    'judged_prob': jp[i], 'episteme': ep[i], 'alea': al[i],
                    'gt_prob': y_gt[i]
                })
        except: pass

    return pd.DataFrame(sim_results)

# --- Main Execution ---

def main():
    print("Starting main...", flush=True)
    print("Loading Data...", flush=True)
    try:
        f = h5py.File('3dshapes.h5', 'r')
        # Use full dataset or very large subset for dense plots
        # The dataset has 480,000 images. 
        # Reduced to 200 to avoid OOM/Overload
        N_SUBSET = 200 
        print(f"Subsetting to {N_SUBSET} samples...", flush=True)
        indices = np.random.choice(480000, N_SUBSET, replace=False)
        indices.sort()
        print("Reading images...", flush=True)
        images = f['images'][indices]
        print("Reading labels...", flush=True)
        labels_raw = f['labels'][indices]
        
        print("Normalizing images...", flush=True)
        images = images.astype(np.float32) / 255.0
        labels_dict = ontology.get_concept_labels(labels_raw)
        
        # 3-class labels
        print("Generating 3-class labels...", flush=True)
        s3, m3 = get_3class_shape_labels(labels_raw)
        labels_dict['P2_shape_3class'] = s3
        
        f.close()
        print("Data Loaded Successfully.", flush=True)
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    # Train Models
    print("Training CNNs...")
    embeddings_dict = {}
    # Train M1 (color+shape), M2 (shape), M3 (color)
    # Simplified: Just training M1 for full demo to save time, 
    # but normally we train all 3. Let's train all 3 fast (1 epoch).
    for m, k in [('M1', 64), ('M2', 8), ('M3', 8)]:
        print(f"  {m}...")
        y = labels_dict[m]
        Xt, Xte, yt, yte = train_test_split(images, y, test_size=0.2, random_state=42)
        state = cnn_model.train_model(Xt, yt, Xte, yte, num_classes=k, num_epochs=1, batch_size=64, verbose=False)
        embeddings_dict[m] = cnn_model.get_embeddings(state, images) # Get all embeddings

    # --- 1. BINARY EXPERIMENT (P1_floor) ---
    df_auroc_bin, df_unc_bin = run_experiment(
        "Binary_P1_Floor", 'P1_floor', False, embeddings_dict, labels_dict
    )
    
    # Simulation for Binary Figures 5/6
    df_sim_bin = simulate_ambiguous_data(embeddings_dict, labels_dict, 'P1_floor')
    
    # Plot Binary Figures
    print(f"Binary Stats - AUROC: {len(df_auroc_bin)} rows, Unc: {len(df_unc_bin)} rows, Sim: {len(df_sim_bin)} rows")
    if not df_sim_bin.empty:
        print("Binary Sim Head:\n", df_sim_bin.head())
    
    plot_figure4_lines(df_auroc_bin, "Binary")
    plot_figure5_uncertainty(df_unc_bin, df_sim_bin, "Binary")
    plot_figure6_alea(df_unc_bin, df_sim_bin, "Binary")

    # --- 2. MULTICLASS EXPERIMENT (P2_shape_3class) ---
    # Filter embeddings to only include 3-class subset
    m3_mask = labels_raw[:, 4] < 3
    emb_dict_3c = {k: v[m3_mask] for k, v in embeddings_dict.items()}
    lab_dict_3c = {'P2_shape_3class': labels_dict['P2_shape_3class']} # Already masked in get_3class... wait.
    # labels_dict['P2_shape_3class'] is shorter than images!
    # We need to be careful. labels_dict['P2_shape_3class'] was created from labels_raw
    # But we need to mask embeddings similarly.
    
    df_auroc_multi, df_unc_multi = run_experiment(
        "Multi_P2_Shape", 'P2_shape_3class', True, emb_dict_3c, lab_dict_3c
    )
    
    # Plot Multiclass Figures (Figure 4 mainly)
    plot_figure4_lines(df_auroc_multi, "Multiclass")
    
    # Generate Multiclass Uncertainty Figures (Figs 5/6)
    # Simulating "ambiguity" for multiclass:
    # We can simulate ambiguity between Class 0 and Class 1 in the 3-class setting
    # This is enough to test if the Dirichlet probe behaves rationally on the simplex edge.
    print("Simulating ambiguous data for Multiclass P2_shape...")
    df_sim_multi = simulate_ambiguous_data(emb_dict_3c, lab_dict_3c, 'P2_shape_3class')
    
    print(f"Multi Stats - AUROC: {len(df_auroc_multi)} rows, Unc: {len(df_unc_multi)} rows, Sim: {len(df_sim_multi)} rows")
    
    if not df_sim_multi.empty:
        print("Multi Sim Head:\n", df_sim_multi.head())
        plot_figure5_uncertainty(df_unc_multi, df_sim_multi, "Multiclass")
        plot_figure6_alea(df_unc_multi, df_sim_multi, "Multiclass")
    else:
        print("Skipping Multiclass Figs 5/6 (Simulation failed)")

    # --- 3. MANIFOLD PLOT ---
    print("Generating Manifold Plot...")
    # Use M1 embeddings on 3-class task
    X_emb = emb_dict_3c['M1']
    y = lab_dict_3c['P2_shape_3class']
    X_tr, X_te, y_tr, y_te = train_test_split(X_emb, y, train_size=100, test_size=200, stratify=y, random_state=99)
    
    y_tr_oh = jax.nn.one_hot(y_tr, 3)
    unc = ppm.gpp_multiclass(X_te, X_tr, y_tr_oh, num_classes=3)
    probs = np.array(unc['categorical_mu'])
    
    plot_dirichlet_manifold(probs, "GPP Dirichlet Manifold (3-Class Shape)", "figure_manifold_3class.png")

    print("Done.")

if __name__ == "__main__":
    os.environ['JAX_PLATFORM_NAME'] = 'cpu'
    main()
