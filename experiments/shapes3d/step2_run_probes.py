import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import gpp_common as common
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm
import jax

# --- CONFIG ---
TEST_MODE = False
TEST_SIZE = 1000
# --------------

def run_experiments():
    print(f"=== Step 2: Run Probing Experiments (TEST_MODE={TEST_MODE}) ===", flush=True)
    common.ensure_dirs()
    
    # Load Data & Embeddings
    try:
        data = np.load(os.path.join(common.EMBEDDING_DIR, 'data_labels.npz'), allow_pickle=True)
        
        # Get test/probe indices
        test_indices = data['test_indices']
        train_indices = data['train_indices'] # NEW: Load training indices for Teacher
        
        if TEST_MODE:
            print(f"TEST MODE: Using only first {TEST_SIZE} test samples.")
            test_indices = test_indices[:TEST_SIZE]
            
        print(f"Using {len(test_indices)} samples for probing (from test set).")

        # Load ALL embeddings first (mapped by index)
        X_emb_M1_full = np.load(os.path.join(common.EMBEDDING_DIR, 'embeddings_M1.npy'))
        X_emb_M2_full = np.load(os.path.join(common.EMBEDDING_DIR, 'embeddings_M2.npy'))
        X_emb_M3_full = np.load(os.path.join(common.EMBEDDING_DIR, 'embeddings_M3.npy'))
        
        # Subset Embeddings for Probing (Test Set)
        embeddings = {
            'M1': X_emb_M1_full[test_indices],
            'M2': X_emb_M2_full[test_indices],
            'M3': X_emb_M3_full[test_indices]
        }
        
        # Teacher Embeddings (Use TRAIN set for better GT estimation)
        print(f"Teacher using {len(train_indices)} training samples for GT estimation.")
        X_emb_teacher_full = X_emb_M1_full[train_indices]

        # Subset Labels
        labels_dict = {}
        labels_dict_train = {} # For Teacher
        
        for k in data.files:
            if k in ['images', 'train_indices', 'val_indices', 'test_indices']: continue
            
            val = data[k]
            # If array matches full dataset length, subset it
            if hasattr(val, 'shape') and val.shape[0] == len(X_emb_M1_full):
                labels_dict[k] = val[test_indices]
                labels_dict_train[k] = val[train_indices] # For Teacher
            else:
                labels_dict[k] = val
                labels_dict_train[k] = val

    except Exception as e:
        print(f"Error loading embeddings: {e}")
        print("Run step1_train_and_embed.py first.")
        return

    # Containers
    all_auroc_bin = pd.DataFrame()
    all_unc_bin = pd.DataFrame()
    all_auroc_multi = pd.DataFrame()
    all_unc_multi = pd.DataFrame()
    all_fuzziness_p1 = pd.DataFrame()
    all_fuzziness_p2 = pd.DataFrame()
    
    # Iterate Models
    for m in ['M1', 'M2', 'M3']:
        print(f"\n--- Probing Model {m} ---", flush=True)
        X_emb = embeddings[m]
        
        # 1. Binary Experiment (P1_floor)
        print(f"Running Binary Experiment (P1_floor)...")
        # Note: X_emb_teacher for binary is just M1 embeddings on TRAIN set
        res_auroc, res_unc = common.run_experiment_single_model(
            "Binary_P1_Floor", 'P1_floor', False, 
            m, X_emb, X_emb_teacher_full, labels_dict['P1_floor'], labels_dict_train['P1_floor']
        )
        all_auroc_bin = pd.concat([all_auroc_bin, res_auroc])
        all_unc_bin = pd.concat([all_unc_bin, res_unc])
        
        # 2. Multiclass Experiment (P2_shape - 3 CLASS SUBSET)
        # Use only 3 classes (0, 1, 2) as requested
        y_shape = labels_dict['P2_shape']
        y_shape_train = labels_dict_train['P2_shape'] # Teacher labels
        
        # Mask for 3 classes
        mask_test = y_shape < 3
        mask_train = y_shape_train < 3
        
        X_emb_3c = X_emb[mask_test]
        y_3c = y_shape[mask_test]
        
        X_emb_teacher_3c = X_emb_teacher_full[mask_train]
        y_3c_train = y_shape_train[mask_train]
        
        if len(y_3c) < 10:
            print("Warning: Not enough samples for 3-class experiment after masking.")
        else:
            print(f"Running Multiclass Experiment (P2_shape) on {len(y_3c)} samples (3 classes)...")
            res_auroc_m, res_unc_m = common.run_experiment_single_model(
                "Multi_P2_Shape", 'P2_shape', True,
                m, X_emb_3c, X_emb_teacher_3c, y_3c, y_3c_train
            )
            
            # Rename GPP to GPP-Dirichlet
            res_auroc_m.loc[res_auroc_m['method'] == 'GPP', 'method'] = 'GPP-Dirichlet'
            res_unc_m.loc[res_unc_m['method'] == 'GPP', 'method'] = 'GPP-Dirichlet'
            
            # Add GPP-Beta (OvR) Trace
            print(f"Running GPP-Beta (OvR) for {m}...")
            # OvR: Class 0 vs Rest
            y_bin = (y_3c == 0).astype(int)
            y_bin_train = (y_3c_train == 0).astype(int)
            
            res_auroc_beta, res_unc_beta = common.run_experiment_single_model(
                "Multi_P2_Shape_BinaryOvR", 'P2_shape', False, 
                m, X_emb_3c, X_emb_teacher_3c, y_bin, y_bin_train
            )
            res_auroc_beta['method'] = 'GPP-Beta'
            res_unc_beta['method'] = 'GPP-Beta'
            
            all_auroc_multi = pd.concat([all_auroc_multi, res_auroc_m, res_auroc_beta])
            all_unc_multi = pd.concat([all_unc_multi, res_unc_m, res_unc_beta])
        
        # 3. Manifold (Only M1)
        # MANIFOLD NEEDS 3 CLASSES for Triangle Plot (Already have 3c data)
        if m == 'M1':
            print("Generating Manifold Data (M1 - 3 Classes)...")
            
            # Use X_emb_3c, y_3c from above
            ts = min(200, int(len(y_3c)*0.5))
            tr_s = min(100, int(len(y_3c)*0.25))
            
            if tr_s > 5 and ts > 5:
                X_tr, X_te, y_tr, y_te = train_test_split(X_emb_3c, y_3c, train_size=tr_s, test_size=ts, stratify=y_3c, random_state=99)
                y_tr_oh = jax.nn.one_hot(y_tr, 3)
                unc = ppm.gpp_multiclass(X_te, X_tr, y_tr_oh, num_classes=3)
                probs = np.array(unc['categorical_mu'])
                
                df_manifold = pd.DataFrame(probs, columns=['P_Class0', 'P_Class1', 'P_Class2'])
                df_manifold.to_csv(os.path.join(common.MANIFOLD_DIR, "manifold_probs.csv"), index=False)
            
            # 4. Fuzziness Experiments (M1)
            print("Running Fuzziness Experiments (M1)...")
            
            # P.1 Fuzziness
            print("  > P.1 (Binary Floor Hue)")
            y_p1 = labels_dict['P1_floor']
            
            df_fuzz_p1 = common.run_fuzziness_experiment(
                task_key='P1_floor', is_multiclass=False, 
                model_name=m, X_emb_model=X_emb, y_all=y_p1, num_classes=2
            )
            if not df_fuzz_p1.empty:
                all_fuzziness_p1 = pd.concat([all_fuzziness_p1, df_fuzz_p1])
                
            # P.2 Fuzziness (Use 3 Classes)
            print("  > P.2 (Multiclass Shape - 3 Classes)")
            df_fuzz_p2 = common.run_fuzziness_experiment(
                task_key='P2_shape', is_multiclass=True, 
                model_name=m, X_emb_model=X_emb_3c, y_all=y_3c, num_classes=3
            )
            if not df_fuzz_p2.empty:
                all_fuzziness_p2 = pd.concat([all_fuzziness_p2, df_fuzz_p2])

    # Save Results
    all_auroc_bin.to_csv(os.path.join(common.CSV_DIR, 'results_jax_auroc_binary.csv'), index=False)
    all_unc_bin.to_csv(os.path.join(common.CSV_DIR, 'results_jax_unc_binary.csv'), index=False)
    all_auroc_multi.to_csv(os.path.join(common.CSV_DIR, 'results_jax_auroc_multi.csv'), index=False)
    all_unc_multi.to_csv(os.path.join(common.CSV_DIR, 'results_jax_unc_multi.csv'), index=False)
    all_fuzziness_p1.to_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p1.csv'), index=False)
    all_fuzziness_p2.to_csv(os.path.join(common.CSV_DIR, 'results_jax_fuzziness_p2.csv'), index=False)
    
    print("\nStep 2 Completed. Results saved to results/csv/")

if __name__ == "__main__":
    run_experiments()
