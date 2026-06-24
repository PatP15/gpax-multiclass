import h5py
import numpy as np
import torch
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.svm as sksvm
from sklearn.model_selection import train_test_split
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

from GPtorch.probing import gp, gp_multiclass
from GPtorch.probing import probabilistic_probe as pp
from GPtorch.probing import probabilistic_probe_multiclass as ppm
import ontology
import cnn_model_torch as cnn_model

# --- Helper Functions ---

def get_3class_shape_labels(labels_raw):
    """Create 3-class shape labels from 4-class (use first 3 shapes: 0,1,2)."""
    shape = labels_raw[:, 4].astype(int)
    mask = shape < 3
    return shape[mask], mask

def compute_lpr_probs(x_query, x_observed, y_observed, num_classes):
    """Linear Probe Regression: Standard Logistic Regression."""
    # Helper for sklm which expects CPU numpy
    if isinstance(x_query, torch.Tensor): x_query = x_query.cpu().numpy()
    if isinstance(x_observed, torch.Tensor): x_observed = x_observed.cpu().numpy()
    if isinstance(y_observed, torch.Tensor): y_observed = y_observed.cpu().numpy()
    
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

def print_gpu_memory():
    try:
        os.system('nvidia-smi | grep "MiB /"')
    except:
        pass

def get_embeddings_batched(model, images, batch_size=2048):
    """Compute embeddings in batches to avoid OOM."""
    num_images = images.shape[0]
    embeddings = []
    print(f"Computing embeddings for {num_images} images (batch_size={batch_size})...")
    
    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size]
        emb = cnn_model.get_embeddings(model, batch)
        embeddings.append(emb)
        if i % (batch_size * 10) == 0:
            print(f"  Processed {i}/{num_images}...", flush=True)
            
    return np.concatenate(embeddings, axis=0)

# --- Plotting Functions (Identical to original) ---

def plot_figure4_lines(df_auroc, scenario_name):
    if df_auroc.empty: return
    fig, axes = plt.subplots(1, 3, figsize=(18, 5), sharey=True)
    models = ['M1', 'M2', 'M3']
    sns.set_style("whitegrid")
    palette = {'GPP': '#1f77b4', 'LPE': '#ff7f0e', 'SVM': '#2ca02c', 'LP': '#d62728', 'Maha': '#9467bd'}
    
    for i, model in enumerate(models):
        ax = axes[i]
        data = df_auroc[df_auroc['model'] == model]
        if data.empty: continue
        sns.lineplot(
            data=data, x='n_obs', y='AUROC', hue='method', style='task_type',
            palette=palette, markers=True, dashes=True, ax=ax, linewidth=2,
            err_style='band', errorbar=('ci', 95)
        )
        ax.set_title(f'Model {model}', fontsize=14)
        ax.set_xlabel('Observations')
        if i == 0: ax.set_ylabel('AUROC')
        else: ax.set_ylabel('')
        ax.set_ylim(0.5, 1.02)
        ax.set_xscale('log')
        ax.set_xticks([2, 8, 32, 128])
        ax.set_xticklabels([2, 8, 32, 128])

    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='upper right', bbox_to_anchor=(0.98, 0.9))
    for ax in axes: ax.get_legend().remove()
    plt.suptitle(f'Figure 4: Learning Curves ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure4_auroc_{scenario_name}_torch.png', dpi=300)
    print(f"Saved figure4_auroc_{scenario_name}_torch.png")

def plot_figure5_uncertainty(df_unc, df_sim, scenario_name):
    fig = plt.figure(figsize=(18, 6))
    ax1 = plt.subplot(1, 3, 1)
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
    
    df_ambiguous = pd.concat([
        df_sim[(df_sim['gt_prob'] >= 0.4) & (df_sim['gt_prob'] <= 0.6)],
        df_unc[(df_unc['gt_prob'] >= 0.4) & (df_unc['gt_prob'] <= 0.6)]
    ])
    
    ax2 = plt.subplot(1, 3, 2)
    d_lpe = df_ambiguous[df_ambiguous['method'] == 'LPE']
    if not d_lpe.empty:
        sns.scatterplot(data=d_lpe, x='episteme', y='judged_prob', hue='n_obs', palette='rocket_r', ax=ax2, legend=False)
        ax2.set_xscale('log')
        ax2.set_title('LPE | GT Prob ≈ 0.5 (Fig 5 Mid)')
    else: ax2.text(0.5, 0.5, 'No Ambiguous Data for LPE')

    ax3 = plt.subplot(1, 3, 3)
    d_gpp = df_ambiguous[df_ambiguous['method'] == 'GPP']
    if not d_gpp.empty:
        sns.scatterplot(data=d_gpp, x='episteme', y='judged_prob', hue='n_obs', palette='rocket_r', ax=ax3)
        ax3.set_xscale('log')
        ax3.set_title('GPP | GT Prob ≈ 0.5 (Fig 5 Right)')
        plt.legend(title='Observations', bbox_to_anchor=(1.05, 1), loc='upper left')
    else: ax3.text(0.5, 0.5, 'No Ambiguous Data for GPP')

    plt.suptitle(f'Figure 5: Uncertainty Analysis ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure5_uncertainty_{scenario_name}_torch.png', dpi=300)
    print(f"Saved figure5_uncertainty_{scenario_name}_torch.png")

def plot_figure6_alea(df_unc, df_sim, scenario_name):
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    df_ambiguous = pd.concat([
        df_sim[(df_sim['gt_prob'] >= 0.4) & (df_sim['gt_prob'] <= 0.6)],
        df_unc[(df_unc['gt_prob'] >= 0.4) & (df_unc['gt_prob'] <= 0.6)]
    ])
    df_confident = df_unc[df_unc['gt_prob'] > 0.9]
    
    ax1 = axes[0]
    d_lpe = df_ambiguous[df_ambiguous['method'] == 'LPE']
    if not d_lpe.empty:
        sns.scatterplot(data=d_lpe, x='episteme', y='alea', hue='n_obs', palette='rocket_r', ax=ax1, legend=False)
        ax1.set_xscale('log')
        ax1.set_title('LPE | GT Prob ≈ 0.5')
    
    ax2 = axes[1]
    d_gpp = df_ambiguous[df_ambiguous['method'] == 'GPP']
    if not d_gpp.empty:
        sns.scatterplot(data=d_gpp, x='episteme', y='alea', hue='n_obs', palette='rocket_r', ax=ax2, legend=False)
        ax2.set_xscale('log')
        ax2.set_title('GPP | GT Prob ≈ 0.5')

    ax3 = axes[2]
    d_gpp_conf = df_confident[df_confident['method'] == 'GPP']
    if not d_gpp_conf.empty:
        sns.scatterplot(data=d_gpp_conf, x='episteme', y='alea', hue='n_obs', palette='rocket_r', ax=ax3)
        ax3.set_xscale('log')
        ax3.set_title('GPP | GT Prob ≈ 1.0')
        plt.legend(title='Observations', bbox_to_anchor=(1.05, 1), loc='upper left')

    plt.suptitle(f'Figure 6: Alea vs Episteme ({scenario_name})', fontsize=16)
    plt.tight_layout()
    plt.savefig(f'figure6_alea_episteme_{scenario_name}_torch.png', dpi=300)
    print(f"Saved figure6_alea_episteme_{scenario_name}_torch.png")

def plot_dirichlet_manifold(probs, title, filename):
    if probs.shape[1] != 3: return
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
    y_all = labels_dict[task_key]
    
    print("Training Teacher Model...")
    X_emb_teacher = embeddings_dict[teacher_model_name]
    L_teach = min(len(X_emb_teacher), len(y_all))
    X_emb_teacher = X_emb_teacher[:L_teach]
    y_teacher = y_all[:L_teach]
    
    X_teach_train, X_teach_test, y_teach_train, y_teach_test = train_test_split(X_emb_teacher, y_teacher, test_size=0.5, random_state=42)
    
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance')
    knn.fit(X_teach_train, y_teach_train)
    gt_probs_test = knn.predict_proba(X_teach_test) 
    
    auroc_results = []
    uncertainty_data = []
    observation_levels = [2, 4, 8, 16, 32, 64, 128]
    repeats_per_level = 5 
    
    for model_name in ['M1', 'M2', 'M3']:
        print(f"  Probing Model: {model_name}")
        X_emb = embeddings_dict[model_name]
        L = min(len(X_emb), len(y_all))
        X_emb = X_emb[:L]
        y = y_all[:L]
        _, X_probe_test, _, y_probe_test = train_test_split(X_emb, y, test_size=0.5, random_state=42)
        
        # Convert probe test data to tensors for GPU acceleration
        if torch.cuda.is_available():
            X_probe_test_torch = torch.tensor(X_probe_test).cuda()
        else:
            X_probe_test_torch = torch.tensor(X_probe_test)

        for n_obs in observation_levels:
            for r in range(repeats_per_level):
                seed = 42 + n_obs * 100 + r
                X_probe_train_pool, _, y_probe_train_pool, _ = train_test_split(X_emb, y, test_size=0.5, random_state=42)
                try:
                    if n_obs >= len(np.unique(y_probe_train_pool)):
                         X_obs, _, y_obs, _ = train_test_split(X_probe_train_pool, y_probe_train_pool, train_size=n_obs, stratify=y_probe_train_pool, random_state=seed)
                    else:
                         indices = np.random.choice(len(X_probe_train_pool), n_obs, replace=False)
                         X_obs = X_probe_train_pool[indices]
                         y_obs = y_probe_train_pool[indices]
                except: continue

                # Convert obs to tensors
                if torch.cuda.is_available():
                    X_obs_torch = torch.tensor(X_obs).cuda()
                    y_obs_torch = torch.tensor(y_obs).cuda() # indices
                else:
                    X_obs_torch = torch.tensor(X_obs)
                    y_obs_torch = torch.tensor(y_obs)

                # 1. GPP
                try:
                    if is_multiclass:
                        # PyTorch one hot
                        y_obs_oh = torch.nn.functional.one_hot(y_obs_torch.long(), gt_probs_test.shape[1]).float()
                        unc = ppm.gpp_multiclass(X_probe_test_torch, X_obs_torch, y_obs_oh, num_classes=gt_probs_test.shape[1])
                        probs = np.array(unc['categorical_mu'].cpu())
                    else:
                        unc = pp.gpp(X_probe_test_torch, X_obs_torch, y_obs_torch.float())
                        probs = np.array(unc['Judged probability'].cpu())
                        episteme = np.array(unc['Episteme'].cpu())
                        alea = np.array(unc['Alea'].cpu())

                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({'model': model_name, 'n_obs': n_obs, 'method': 'GPP', 'AUROC': score, 'task_type': scenario_name})
                    
                    if not is_multiclass:
                        for i in range(min(len(probs), 50)):
                            uncertainty_data.append({
                                'model': model_name, 'n_obs': n_obs, 'method': 'GPP',
                                'judged_prob': float(probs[i]),
                                'episteme': float(episteme[i]),
                                'alea': float(alea[i]),
                                'gt_prob': float(gt_probs_test[i, 1])
                            })
                except Exception as e: pass # print(f"GPP Error: {e}")

                # 2. LPE
                try:
                    # LPE handles numpy internally if needed or tensors
                    if is_multiclass:
                        unc = ppm.lpe_multiclass(X_probe_test_torch, X_obs_torch, y_obs_torch, repeats=20, num_classes=gt_probs_test.shape[1])
                        probs = np.array(unc['categorical_mu'].cpu())
                    else:
                        unc = pp.lpe(X_probe_test_torch, X_obs_torch, y_obs_torch, repeats=20)
                        probs = np.array(unc['Judged probability'].cpu())
                        episteme = np.array(unc['Episteme'].cpu())
                        alea = np.array(unc['Alea'].cpu())

                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({'model': model_name, 'n_obs': n_obs, 'method': 'LPE', 'AUROC': score, 'task_type': scenario_name})
                    
                    if not is_multiclass:
                         for i in range(min(len(probs), 50)):
                            uncertainty_data.append({
                                'model': model_name, 'n_obs': n_obs, 'method': 'LPE',
                                'judged_prob': float(probs[i]),
                                'episteme': float(episteme[i]),
                                'alea': float(alea[i]),
                                'gt_prob': float(gt_probs_test[i, 1])
                            })
                except Exception as e: pass # print(f"LPE Error: {e}")

                # 3. SVM (CPU based, use numpy)
                try:
                    if is_multiclass:
                        clf = sksvm.SVC(kernel='linear', probability=True)
                        clf.fit(X_obs, y_obs)
                        probs = clf.predict_proba(X_probe_test)
                    else:
                        # probabilistic_probe.svm_probe takes torch/numpy
                        res = pp.svm_probe(X_probe_test_torch, X_obs_torch, y_obs_torch)
                        probs = np.array(res['Judged probability'].cpu())
                    
                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({'model': model_name, 'n_obs': n_obs, 'method': 'SVM', 'AUROC': score, 'task_type': scenario_name})
                except: pass

                # 4. LP
                try:
                    probs = compute_lpr_probs(X_probe_test, X_obs, y_obs, gt_probs_test.shape[1])
                    score = compute_auroc(y_probe_test, probs, is_multiclass)
                    auroc_results.append({'model': model_name, 'n_obs': n_obs, 'method': 'LP', 'AUROC': score, 'task_type': scenario_name})
                except: pass

    return pd.DataFrame(auroc_results), pd.DataFrame(uncertainty_data)

def simulate_ambiguous_data(embeddings_dict, labels_dict, task_key):
    print(f"Simulating ambiguous data for {task_key}...")
    X_emb = np.array(embeddings_dict['M1'])
    y_full = labels_dict[task_key]
    L = min(len(X_emb), len(y_full))
    X = X_emb[:L]
    y = y_full[:L]
    
    c0 = X[y==0].mean(axis=0)
    c1 = X[y==1].mean(axis=0)
    alphas = np.random.uniform(0, 1, 5000)
    X_sim = np.array([(1-a)*c0 + a*c1 for a in alphas])
    y_gt = alphas
    
    if torch.cuda.is_available():
        X_sim_torch = torch.tensor(X_sim).cuda()
    else:
        X_sim_torch = torch.tensor(X_sim)
    
    sim_results = []
    if len(np.unique(y)) < 2: return pd.DataFrame()

    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    pool_size = 200 
    if len(idx0) < pool_size or len(idx1) < pool_size: 
         pool_size = min(len(idx0), len(idx1))
         if pool_size < 10: return pd.DataFrame()
    
    idx_train = np.concatenate([
        np.random.choice(idx0, pool_size, replace=False),
        np.random.choice(idx1, pool_size, replace=False)
    ])
    X_train = X[idx_train]
    y_train = y[idx_train]
    is_multiclass_mode = (len(np.unique(y_full)) > 2)

    for n_obs in [2, 8, 32, 128]:
        if n_obs > len(X_train): continue
        idx_sub = np.random.choice(len(X_train), n_obs, replace=False)
        X_tr_sub = X_train[idx_sub]
        y_tr_sub = y_train[idx_sub]
        
        if torch.cuda.is_available():
            X_tr_sub_torch = torch.tensor(X_tr_sub).cuda()
            y_tr_sub_torch = torch.tensor(y_tr_sub).cuda()
        else:
            X_tr_sub_torch = torch.tensor(X_tr_sub)
            y_tr_sub_torch = torch.tensor(y_tr_sub)
        
        try:
            if is_multiclass_mode:
                K = 3
                y_tr_oh = torch.nn.functional.one_hot(y_tr_sub_torch.long(), K).float()
                unc = ppm.gpp_multiclass(X_sim_torch, X_tr_sub_torch, y_tr_oh, num_classes=K)
                probs_all = np.array(unc['categorical_mu'].cpu())
                jp = probs_all[:, 1] 
                ep = np.max(probs_all, axis=1)
                al = np.zeros_like(jp) 
            else:
                unc = pp.gpp(X_sim_torch, X_tr_sub_torch, y_tr_sub_torch.float())
                jp = np.array(unc['Judged probability'].cpu()).flatten()
                ep = np.array(unc['Episteme'].cpu()).flatten()
                al = np.array(unc['Alea'].cpu()).flatten()
                
            for i in range(len(X_sim)):
                sim_results.append({
                    'method': 'GPP', 'n_obs': n_obs, 
                    'judged_prob': jp[i], 'episteme': ep[i], 'alea': al[i], 'gt_prob': y_gt[i]
                })
        except Exception as e: pass # print(f"Sim Error: {e}")
        
        try:
            if is_multiclass_mode:
                unc = ppm.lpe_multiclass(X_sim_torch, X_tr_sub_torch, y_tr_sub_torch, repeats=20, num_classes=3)
                probs_all = np.array(unc['categorical_mu'].cpu())
                jp = probs_all[:, 1]
                ep = np.max(probs_all, axis=1)
                al = np.zeros_like(jp)
            else:
                unc = pp.lpe(X_sim_torch, X_tr_sub_torch, y_tr_sub_torch)
                jp = np.array(unc['Judged probability'].cpu()).flatten()
                ep = np.array(unc['Episteme'].cpu()).flatten()
                al = np.array(unc['Alea'].cpu()).flatten()
                
            for i in range(len(X_sim)):
                sim_results.append({
                    'method': 'LPE', 'n_obs': n_obs, 
                    'judged_prob': jp[i], 'episteme': ep[i], 'alea': al[i], 'gt_prob': y_gt[i]
                })
        except: pass

    return pd.DataFrame(sim_results)

def main():
    print("Starting main (PyTorch Version)...", flush=True)
    print_gpu_memory()
    print("Loading Data...", flush=True)
    try:
        f = h5py.File('3dshapes.h5', 'r')
        # N_SUBSET = 5000 # Reduced for quick demo, set to None for full
        N_SUBSET = 5000
        if N_SUBSET:
            indices = np.random.choice(480000, N_SUBSET, replace=False)
            indices.sort()
            images = f['images'][indices]
            labels_raw = f['labels'][indices]
        else:
            images = f['images'][:]
            labels_raw = f['labels'][:]
        
        labels_dict = ontology.get_concept_labels(labels_raw)
        s3, m3 = get_3class_shape_labels(labels_raw)
        labels_dict['P2_shape_3class'] = s3
        f.close()
    except Exception as e:
        print(f"Data Load Error: {e}")
        return

    print("Training CNNs (PyTorch)...")
    embeddings_dict = {}
    # Train M1, M2, M3
    for m, k in [('M1', 64), ('M2', 8), ('M3', 8)]:
        print(f"  {m}...")
        y = labels_dict[m]
        Xt, Xte, yt, yte = train_test_split(images, y, test_size=0.2, random_state=42)
        # Only 1 epoch to be fast as in original
        model = cnn_model.train_model(Xt, yt, Xte, yte, num_classes=k, num_epochs=1, batch_size=64, verbose=False)
        embeddings_dict[m] = get_embeddings_batched(model, images)

    # 1. BINARY
    df_auroc_bin, df_unc_bin = run_experiment("Binary_P1_Floor", 'P1_floor', False, embeddings_dict, labels_dict)
    
    # Save Binary Results
    df_auroc_bin.to_csv('results_torch_auroc_binary.csv', index=False)
    df_unc_bin.to_csv('results_torch_unc_binary.csv', index=False)

    df_sim_bin = simulate_ambiguous_data(embeddings_dict, labels_dict, 'P1_floor')
    if not df_sim_bin.empty:
        df_sim_bin.to_csv('results_torch_sim_binary.csv', index=False)

    plot_figure4_lines(df_auroc_bin, "Binary")
    plot_figure5_uncertainty(df_unc_bin, df_sim_bin, "Binary")
    plot_figure6_alea(df_unc_bin, df_sim_bin, "Binary")

    # 2. MULTICLASS
    m3_mask = labels_raw[:, 4] < 3
    emb_dict_3c = {k: v[m3_mask] for k, v in embeddings_dict.items()}
    lab_dict_3c = {'P2_shape_3class': labels_dict['P2_shape_3class']}
    
    df_auroc_multi, df_unc_multi = run_experiment("Multi_P2_Shape", 'P2_shape_3class', True, emb_dict_3c, lab_dict_3c)
    
    # Save Multiclass Results
    df_auroc_multi.to_csv('results_torch_auroc_multi.csv', index=False)
    df_unc_multi.to_csv('results_torch_unc_multi.csv', index=False)

    plot_figure4_lines(df_auroc_multi, "Multiclass")
    
    print("Simulating ambiguous data for Multiclass...")
    df_sim_multi = simulate_ambiguous_data(emb_dict_3c, lab_dict_3c, 'P2_shape_3class')
    if not df_sim_multi.empty:
        df_sim_multi.to_csv('results_torch_sim_multi.csv', index=False)
        plot_figure5_uncertainty(df_unc_multi, df_sim_multi, "Multiclass")
        plot_figure6_alea(df_unc_multi, df_sim_multi, "Multiclass")

    # 3. MANIFOLD
    print("Generating Manifold Plot...")
    X_emb = emb_dict_3c['M1']
    y = lab_dict_3c['P2_shape_3class']
    X_tr, X_te, y_tr, y_te = train_test_split(X_emb, y, train_size=100, test_size=200, stratify=y, random_state=99)
    
    if torch.cuda.is_available():
        X_te_torch = torch.tensor(X_te).cuda()
        X_tr_torch = torch.tensor(X_tr).cuda()
        y_tr_torch = torch.tensor(y_tr).cuda()
    else:
        X_te_torch = torch.tensor(X_te)
        X_tr_torch = torch.tensor(X_tr)
        y_tr_torch = torch.tensor(y_tr)
        
    y_tr_oh = torch.nn.functional.one_hot(y_tr_torch.long(), 3).float()
    unc = ppm.gpp_multiclass(X_te_torch, X_tr_torch, y_tr_oh, num_classes=3)
    probs = np.array(unc['categorical_mu'].cpu())
    
    plot_dirichlet_manifold(probs, "GPP Dirichlet Manifold (3-Class Shape)", "figure_manifold_3class_torch.png")
    print("Done.")

if __name__ == "__main__":
    main()

