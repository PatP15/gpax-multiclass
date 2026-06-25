import os
import sys


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

import numpy as np
import jax
import jax.numpy as jnp
try:
    import h5py  # only used by step1 (raw 3dshapes.h5); step2/step3 read .npy/.npz
except ImportError:
    h5py = None
import sklearn.linear_model as sklm
import sklearn.metrics as skm
import sklearn.svm as sksvm
import pandas as pd
import ontology
import cnn_model
from GPax.probing import gp, gp_multiclass
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm

# --- Constants & Config ---
DATA_PATH = os.path.join(_REPO_ROOT, '3dshapes.h5')
EMBEDDING_DIR = os.path.join(_REPO_ROOT, 'results', 'embeddings')
CSV_DIR = os.path.join(_REPO_ROOT, 'results', 'csv')
FIGURE_DIR = os.path.join(_REPO_ROOT, 'results', 'figures')
MANIFOLD_DIR = os.path.join(_REPO_ROOT, 'results', 'results_manifold')

def ensure_dirs():
    """Create necessary directories."""
    for d in [EMBEDDING_DIR, CSV_DIR, FIGURE_DIR, MANIFOLD_DIR]:
        os.makedirs(d, exist_ok=True)

# --- Data Loading ---

def load_data(subset_size=None):
    """Load 3DShapes data."""
    print("Loading Data...", flush=True)
    try:
        f = h5py.File(DATA_PATH, 'r')
        if subset_size:
            print(f"Subsetting to {subset_size} samples...", flush=True)
            indices = np.random.choice(480000, subset_size, replace=False)
            indices.sort()
            images = f['images'][indices]
            labels_raw = f['labels'][indices]
        else:
            print("Using FULL dataset...", flush=True)
            images = f['images'][:]
            labels_raw = f['labels'][:]
        f.close()
        
        # Process labels
        labels_dict = ontology.get_concept_labels(labels_raw)
        
        # Add 3-class shape labels for P2
        s3, m3 = get_3class_shape_labels(labels_raw)
        labels_dict['P2_shape_3class'] = s3
        labels_dict['mask_3class'] = m3
        
        return images, labels_raw, labels_dict
        
    except Exception as e:
        print(f"Error loading data: {e}")
        sys.exit(1)

def get_3class_shape_labels(labels_raw):
    """Create 3-class shape labels from 4-class (use first 3 shapes: 0,1,2)."""
    shape = labels_raw[:, 4].astype(int)
    mask = shape < 3
    return shape[mask], mask

# --- Model Helpers ---

def get_embeddings_batched(state, images, batch_size=2048):
    """Compute embeddings in batches to avoid OOM."""
    num_images = images.shape[0]
    embeddings = []
    for i in range(0, num_images, batch_size):
        batch = images[i:i+batch_size]
        emb = cnn_model.get_embeddings(state, batch)
        embeddings.append(np.array(emb))
    return np.concatenate(embeddings, axis=0)

def save_checkpoint(state, workdir, model_name):
    """Save flax checkpoint."""
    from flax.training import checkpoints
    # Ensure absolute path to avoid Orbax ValueError
    workdir = os.path.abspath(workdir)
    step = int(state.step)
    checkpoints.save_checkpoint(ckpt_dir=workdir, target=state, step=step, prefix=f"checkpoint_{model_name}_", keep=1, overwrite=True)

def load_checkpoint(state, workdir, model_name):
    """Load flax checkpoint."""
    from flax.training import checkpoints
    # Ensure absolute path to avoid Orbax ValueError
    workdir = os.path.abspath(workdir)
    return checkpoints.restore_checkpoint(ckpt_dir=workdir, target=state, prefix=f"checkpoint_{model_name}_")

# --- Probing Helpers ---

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

def simulate_ambiguous_data(embeddings_dict, labels_dict, task_key):
    """Generate synthetic ambiguous data by interpolating centroids."""
    print(f"Simulating ambiguous data for {task_key}...")
    # Handle both dict of embeddings and single array if passed
    if 'M1' in embeddings_dict:
        X_emb = np.array(embeddings_dict['M1'])
    else:
        X_emb = np.array(list(embeddings_dict.values())[0])
        
    y_full = labels_dict[task_key]
    
    # Convert to standard numpy
    if hasattr(y_full, 'tolist'): y_full = np.array(y_full)
    
    # Truncate
    L = min(len(X_emb), len(y_full))
    X = X_emb[:L]
    y = y_full[:L]
    
    # Centroids
    c0 = X[y==0].mean(axis=0)
    c1 = X[y==1].mean(axis=0)
    
    # Interpolate with more points for denser plot
    alphas = np.random.uniform(0, 1, 5000)
    X_sim = np.array([(1-a)*c0 + a*c1 for a in alphas])
    y_gt = alphas # GT Probability of class 1
    
    # Run Probes on Simulation
    sim_results = []
    
    # We need a training set to probe AGAINST.
    if len(np.unique(y)) < 2:
        print(f"Warning: Only 1 class in simulation data for {task_key}. Skipping sim.")
        return pd.DataFrame()

    idx0 = np.where(y==0)[0]
    idx1 = np.where(y==1)[0]
    
    pool_size = 200 
    if len(idx0) < pool_size or len(idx1) < pool_size: 
         pool_size = min(len(idx0), len(idx1))
         if pool_size < 10:
             return pd.DataFrame()
    
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
        
        # GPP
        try:
            if is_multiclass_mode:
                K = 3
                y_tr_oh = jax.nn.one_hot(y_tr_sub, K)
                unc = ppm.gpp_multiclass(X_sim, X_tr_sub, y_tr_oh, num_classes=K)
                probs_all = np.array(unc['categorical_mu'])
                jp = probs_all[:, 1] 
                ep = np.array(unc['Episteme']).flatten()
                al = np.array(unc['Alea']).flatten()
            else:
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
        except: pass
        
        # LPE
        try:
            if is_multiclass_mode:
                unc = ppm.lpe_multiclass(X_sim, X_tr_sub, y_tr_sub, repeats=20)
                probs_all = np.array(unc['categorical_mu'])
                jp = probs_all[:, 1]
                ep = np.array(unc['Episteme']).flatten()
                al = np.array(unc['Alea']).flatten()
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

def run_fuzziness_experiment(task_key, is_multiclass, model_name, X_emb_model, y_all, num_classes):
    """
    Runs 'Concept Fuzziness' experiment.
    Controls fuzziness by flipping positive labels to negative/random in training data.
    Tests Ground Truth Probabilities: [0.25, 0.5, 0.75, 1.0].
    """
    print(f"\n>>> Experiment: Fuzziness | {task_key} | {model_name}")
    
    X_emb = X_emb_model
    L = min(len(X_emb), len(y_all))
    X_emb = X_emb[:L]
    y = y_all[:L]
    
    # Split Train/Test
    # We use a fixed test set for evaluation (Queries)
    from sklearn.model_selection import train_test_split
    X_train_pool, X_test, y_train_pool, y_test = train_test_split(
        X_emb, y, test_size=0.2, random_state=42, stratify=y
    )
    
    results_list = []
    
    gt_probs = [0.25, 0.5, 0.75, 1.0]
    obs_levels = [2, 8, 32, 128]
    
    for p in gt_probs:
        print(f"  GT Prob: {p}")
        
        # Generate Noisy Training Data
        # Logic: For each sample in X_train_pool, keep label with prob p, flip with 1-p.
        # Flip logic:
        # Binary (P1): If y=1, keep 1 w/ prob p, flip to 0. If y=0, keep 0.
        
        y_train_noisy = y_train_pool.copy()
        rng = np.random.RandomState(42 + int(p*100))
        
        if not is_multiclass: # Binary P1
            # Flip 1 -> 0 with prob (1-p)
            pos_indices = np.where(y_train_pool == 1)[0]
            n_flip = int(len(pos_indices) * (1-p))
            if n_flip > 0:
                flip_idx = rng.choice(pos_indices, n_flip, replace=False)
                y_train_noisy[flip_idx] = 0
        else: # Multiclass P2
            # Let's apply noise to Class 0 ("Square") to make it fuzzy.
            # Flip 0 -> Random(1, 2) with prob (1-p)
            pos_indices = np.where(y_train_pool == 0)[0]
            n_flip = int(len(pos_indices) * (1-p))
            if n_flip > 0:
                flip_idx = rng.choice(pos_indices, n_flip, replace=False)
                # Pick new label from [1, 2] randomly
                new_labels = rng.choice([1, 2], size=n_flip)
                y_train_noisy[flip_idx] = new_labels
        
        # Loop over observation levels
        for n_obs in obs_levels:
            # Sample n_obs training points from this noisy pool
            try:
                idx_obs = rng.choice(len(X_train_pool), n_obs, replace=False)
                X_obs = X_train_pool[idx_obs]
                y_obs = y_train_noisy[idx_obs]
            except: continue
            
            # --- Evaluation ---
            # We evaluate on X_test (Original labels).
            gt_prob_vec = np.zeros(len(X_test))
            
            target_class = 1 if not is_multiclass else 0
            
            for i, true_label in enumerate(y_test):
                if true_label == target_class:
                    gt_prob_vec[i] = p
                else:
                    gt_prob_vec[i] = 0.0
            
            # Now run probes
            
            # 1. LPE
            try:
                if is_multiclass:
                    unc = ppm.lpe_multiclass(X_test, X_obs, y_obs, repeats=10)
                    probs = np.array(unc['categorical_mu'])[:, target_class] # Prob of target
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                else:
                    unc = pp.lpe(X_test, X_obs, y_obs)
                    probs = np.array(unc['Judged probability']).flatten()
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                    
                for i in range(len(X_test)):
                    results_list.append({
                        'task': task_key, 'method': 'LPE', 'gt_prob': gt_prob_vec[i],
                        'judged_prob': probs[i], 'episteme': ep[i], 'alea': al[i],
                        'n_obs': n_obs
                    })
            except Exception as e: print(e)
    
            # 2. GPP-Beta (Standard Binary or OvR for Multi)
            try:
                if is_multiclass:
                    # OvR: Train binary GPP on (Class 0 vs Rest) using y_obs
                    y_obs_bin = (y_obs == target_class).astype(int)
                    unc = pp.gpp(X_test, X_obs, y_obs_bin)
                    probs = np.array(unc['Judged probability']).flatten()
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                else:
                    unc = pp.gpp(X_test, X_obs, y_obs)
                    probs = np.array(unc['Judged probability']).flatten()
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                    
                for i in range(len(X_test)):
                    results_list.append({
                        'task': task_key, 'method': 'GPP-Beta', 'gt_prob': gt_prob_vec[i],
                        'judged_prob': probs[i], 'episteme': ep[i], 'alea': al[i],
                        'n_obs': n_obs
                    })
            except Exception as e: print(e)
    
            # 3. GPP-Dirichlet (Only for Multiclass, or Binary treated as Multi)
            try:
                if is_multiclass:
                    y_obs_oh = jax.nn.one_hot(y_obs, num_classes)
                    unc = ppm.gpp_multiclass(X_test, X_obs, y_obs_oh, num_classes=num_classes)
                    probs = np.array(unc['categorical_mu'])[:, target_class]
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                    
                    for i in range(len(X_test)):
                        results_list.append({
                            'task': task_key, 'method': 'GPP-Dirichlet', 'gt_prob': gt_prob_vec[i],
                            'judged_prob': probs[i], 'episteme': ep[i], 'alea': al[i],
                            'n_obs': n_obs
                        })
                else:
                    # For Binary P1, can run Dirichlet 2-class?
                    # Yes, treating 0/1 as 2 classes.
                    y_obs_oh = jax.nn.one_hot(y_obs, 2)
                    unc = ppm.gpp_multiclass(X_test, X_obs, y_obs_oh, num_classes=2)
                    probs = np.array(unc['categorical_mu'])[:, 1] # Class 1
                    ep = np.array(unc['Episteme']).flatten()
                    al = np.array(unc['Alea']).flatten()
                    
                    for i in range(len(X_test)):
                        results_list.append({
                            'task': task_key, 'method': 'GPP-Dirichlet', 'gt_prob': gt_prob_vec[i],
                            'judged_prob': probs[i], 'episteme': ep[i], 'alea': al[i],
                            'n_obs': n_obs
                        })
            except Exception as e: print(e)

    return pd.DataFrame(results_list)

def run_experiment_single_model(scenario_name, task_key, is_multiclass, 
                              model_name, X_emb_model, 
                              X_emb_teacher, y_all, y_teacher):
    """Runs probe experiment for ONE model to save memory."""
    print(f"\n>>> Experiment: {scenario_name} | Model: {model_name}")
    
    # 1. Setup Teacher (GT estimation)
    # Teacher uses X_emb_teacher and y_teacher (which are typically from the TRAINING set)
    # X_emb_model and y_all are from the TEST set (to be probed)
    
    # Align labels
    L_teach = min(len(X_emb_teacher), len(y_teacher))
    X_teach = X_emb_teacher[:L_teach]
    y_teach = y_teacher[:L_teach]
    
    # Subsample Teacher for KNN efficiency if too large
    # We use ALL available teacher data (no split needed since we provided explicit train set)
    MAX_KNN_SAMPLES = 20000
    if len(X_teach) > MAX_KNN_SAMPLES:
        print(f"  Subsampling Teacher KNN from {len(X_teach)} to {MAX_KNN_SAMPLES}...")
        idx_sub = np.random.RandomState(42).choice(len(X_teach), MAX_KNN_SAMPLES, replace=False)
        X_knn_train = X_teach[idx_sub]
        y_knn_train = y_teach[idx_sub]
    else:
        X_knn_train = X_teach
        y_knn_train = y_teach
        
    # KNN for GT
    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=20, weights='distance', n_jobs=-1) # Use n_jobs=-1 for parallel
    knn.fit(X_knn_train, y_knn_train)
    
    # 2. Process THIS Model (Probe on TEST set)
    X_emb = X_emb_model
    L = min(len(X_emb), len(y_all))
    X_emb = X_emb[:L]
    y = y_all[:L]
    
    # We split the TEST set 50/50 into Probe Train (Observed) and Probe Test (Evaluation)
    from sklearn.model_selection import train_test_split
    X_probe_train_pool, X_probe_test, y_probe_train_pool, y_probe_test = train_test_split(X_emb, y, test_size=0.5, random_state=42)
    
    # Batch prediction of GT probabilities on X_probe_test
    print(f"  Predicting GT probabilities for {len(X_probe_test)} evaluation samples...")
    # Note: We predict GT for the samples we EVALUATE on (X_probe_test)
    # The teacher logic was correct before, but now we are explicit.
    # Before: X_teach_test was effectively X_probe_test (since both were from the same pool).
    # Now: We explicitly use X_probe_test.
    
    gt_probs_test_list = []
    batch_size_knn = 5000
    for i in range(0, len(X_probe_test), batch_size_knn):
        batch = X_probe_test[i:i+batch_size_knn]
        gt_probs_test_list.append(knn.predict_proba(batch))
    gt_probs_test = np.concatenate(gt_probs_test_list, axis=0)
    
    # Metric Containers
    auroc_results = []
    uncertainty_data = []
    
    # Probing Config
    observation_levels = [2, 4, 8, 16, 32, 64, 128]
    repeats_per_level = 5
    
    # Loop levels
    for n_obs in observation_levels:
        for r in range(repeats_per_level):
            seed = 42 + n_obs * 100 + r
            
            # Sample Training Data from Probe Train Pool
            try:
                if n_obs >= len(np.unique(y_probe_train_pool)):
                     X_obs, _, y_obs, _ = train_test_split(
                        X_probe_train_pool, y_probe_train_pool, 
                        train_size=n_obs, stratify=y_probe_train_pool, random_state=seed)
                else:
                     indices = np.random.choice(len(X_probe_train_pool), n_obs, replace=False)
                     X_obs = X_probe_train_pool[indices]
                     y_obs = y_probe_train_pool[indices]
            except: continue

            # --- RUN PROBES ---
            
            # 1. GPP
            try:
                if is_multiclass:
                    y_obs_oh = jax.nn.one_hot(y_obs, gt_probs_test.shape[1])
                    
                    # BATCHED GPP
                    batch_size_probe = 2000
                    probs_batches = []
                    ep_batches = []
                    al_batches = []
                    for i in range(0, len(X_probe_test), batch_size_probe):
                        batch_x = X_probe_test[i:i+batch_size_probe]
                        unc_batch = ppm.gpp_multiclass(batch_x, X_obs, y_obs_oh, num_classes=gt_probs_test.shape[1])
                        probs_batches.append(np.array(unc_batch['categorical_mu']))
                        ep_batches.append(np.array(unc_batch['Episteme']).flatten())
                        al_batches.append(np.array(unc_batch['Alea']).flatten())
                    probs = np.concatenate(probs_batches, axis=0)
                    episteme = np.concatenate(ep_batches, axis=0)
                    alea = np.concatenate(al_batches, axis=0)
                else:
                    # BATCHED GPP BINARY
                    batch_size_probe = 2000
                    probs_batches = []
                    ep_batches = []
                    al_batches = []
                    for i in range(0, len(X_probe_test), batch_size_probe):
                        batch_x = X_probe_test[i:i+batch_size_probe]
                        unc_batch = pp.gpp(batch_x, X_obs, y_obs)
                        probs_batches.append(np.array(unc_batch['Judged probability']))
                        ep_batches.append(np.array(unc_batch['Episteme']))
                        al_batches.append(np.array(unc_batch['Alea']))
                    probs = np.concatenate(probs_batches, axis=0)
                    episteme = np.concatenate(ep_batches, axis=0)
                    alea = np.concatenate(al_batches, axis=0)

                score = compute_auroc(y_probe_test, probs, is_multiclass)
                auroc_results.append({
                    'model': model_name, 'n_obs': n_obs, 'method': 'GPP', 
                    'AUROC': score, 'task_type': scenario_name
                })
                
                if not is_multiclass:
                    for i in range(min(len(probs), 50)):
                        uncertainty_data.append({
                            'model': model_name, 'n_obs': n_obs, 'method': 'GPP',
                            'judged_prob': float(probs[i]),
                            'episteme': float(episteme[i]),
                            'alea': float(alea[i]),
                            'gt_prob': float(gt_probs_test[i, 1])
                        })
                else:
                     max_probs = np.max(probs, axis=1)
                     max_gt = np.max(gt_probs_test, axis=1)
                     for i in range(min(len(probs), 50)):
                        uncertainty_data.append({
                            'model': model_name, 'n_obs': n_obs, 'method': 'GPP',
                            'judged_prob': float(max_probs[i]),
                            'episteme': float(episteme[i]),
                            'alea': float(alea[i]),
                            'gt_prob': float(max_gt[i])
                        })
            except Exception as e: pass

            # 2. LPE
            try:
                if is_multiclass:
                    batch_size_probe = 2000
                    probs_batches = []
                    ep_batches = []
                    al_batches = []
                    for i in range(0, len(X_probe_test), batch_size_probe):
                        batch_x = X_probe_test[i:i+batch_size_probe]
                        unc_batch = ppm.lpe_multiclass(batch_x, X_obs, y_obs, repeats=20)
                        probs_batches.append(np.array(unc_batch['categorical_mu']))
                        ep_batches.append(np.array(unc_batch['Episteme']).flatten())
                        al_batches.append(np.array(unc_batch['Alea']).flatten())
                    probs = np.concatenate(probs_batches, axis=0)
                    episteme = np.concatenate(ep_batches, axis=0)
                    alea = np.concatenate(al_batches, axis=0)
                else:
                    batch_size_probe = 2000
                    probs_batches = []
                    ep_batches = []
                    al_batches = []
                    for i in range(0, len(X_probe_test), batch_size_probe):
                        batch_x = X_probe_test[i:i+batch_size_probe]
                        unc_batch = pp.lpe(batch_x, X_obs, y_obs, repeats=20)
                        probs_batches.append(np.array(unc_batch['Judged probability']))
                        ep_batches.append(np.array(unc_batch['Episteme']))
                        al_batches.append(np.array(unc_batch['Alea']))
                    probs = np.concatenate(probs_batches, axis=0)
                    episteme = np.concatenate(ep_batches, axis=0)
                    alea = np.concatenate(al_batches, axis=0)

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
                else:
                     max_probs = np.max(probs, axis=1)
                     max_gt = np.max(gt_probs_test, axis=1)
                     for i in range(min(len(probs), 50)):
                        uncertainty_data.append({
                            'model': model_name, 'n_obs': n_obs, 'method': 'LPE',
                            'judged_prob': float(max_probs[i]),
                            'episteme': float(episteme[i]),
                            'alea': float(alea[i]),
                            'gt_prob': float(max_gt[i])
                        })
            except: pass

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

            # 4. LP
            try:
                probs = compute_lpr_probs(X_probe_test, X_obs, y_obs, gt_probs_test.shape[1])
                score = compute_auroc(y_probe_test, probs, is_multiclass)
                auroc_results.append({
                    'model': model_name, 'n_obs': n_obs, 'method': 'LP', 
                    'AUROC': score, 'task_type': scenario_name
                })
            except: pass
            
    return pd.DataFrame(auroc_results), pd.DataFrame(uncertainty_data)
