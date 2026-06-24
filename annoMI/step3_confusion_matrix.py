import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
import argparse

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import annomi_common as common

def plot_confusion_matrix(y_true, y_pred, classes, title, save_path):
    cm = confusion_matrix(y_true, y_pred)
    # Normalize
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm_norm, annot=True, fmt='.2f', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.title(title)
    plt.ylabel('True Label')
    plt.xlabel('Predicted Label')
    plt.tight_layout()
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved confusion matrix to {save_path}")

def main():
    parser = argparse.ArgumentParser(description="Generate Confusion Matrices for AnnoMI Experiments")
    parser.add_argument("--n", type=int, default=None, help="Sample size to analyze (defaults to max available)")
    args = parser.parse_args()

    print(f"Generating Confusion Matrices for Model: {common.MODEL_NAME} ({common.MODEL_TYPE})")
    
    # Experiments to check
    experiments = {
        'Client Only': 'raw_exp1.npz',
        'Context': 'raw_exp2.npz',
        'Context (Prompted)': 'raw_exp2_prompted.npz',
        'Cascading': 'raw_exp4.npz', # Note: Exp4 might not save raw data in the same way? Let's check.
        'LPE': 'raw_lpe.npz'
    }
    
    classes = ['Sustain', 'Neutral', 'Change']
    
    for exp_name, filename in experiments.items():
        fpath = os.path.join(common.DATA_DIR, filename)
        if not os.path.exists(fpath):
            print(f"Skipping {exp_name}: File not found ({fpath})")
            continue
            
        try:
            data = np.load(fpath)
            
            # Find available n
            available_ns = set()
            for key in data.files:
                if key.startswith('n') and '_y_true' in key:
                    n_str = key.split('_')[0][1:] # 'n100' -> '100'
                    if n_str.isdigit():
                        available_ns.add(int(n_str))
            
            if not available_ns:
                print(f"No valid data found in {filename}")
                continue
                
            # Select n
            target_n = args.n if args.n else max(available_ns)
            if target_n not in available_ns:
                print(f"Sample size {target_n} not found for {exp_name}. Available: {sorted(list(available_ns))}")
                continue
                
            print(f"Processing {exp_name} (n={target_n})...")
            
            y_true = data[f'n{target_n}_y_true']
            y_pred = data[f'n{target_n}_y_pred']

            # --- Balance the Test Set for Visualization ---
            # Undersample majority classes in the test set to match minority class
            # This is ONLY for visualization/metrics calculated here, does not affect stored results
            unique_classes, counts = np.unique(y_true, return_counts=True)
            min_count = np.min(counts)
            print(f"  Balancing test set for CM: Reducing all classes to {min_count} samples.")
            
            balanced_indices = []
            rng = np.random.default_rng(42) # Fixed seed for visualization stability
            
            for cls in unique_classes:
                cls_indices = np.where(y_true == cls)[0]
                if len(cls_indices) > min_count:
                    selected = rng.choice(cls_indices, size=min_count, replace=False)
                else:
                    selected = cls_indices
                balanced_indices.append(selected)
                
            all_balanced_indices = np.concatenate(balanced_indices)
            # No need to shuffle for confusion matrix, but good practice
            # y_true = y_true[all_balanced_indices]
            # y_pred = y_pred[all_balanced_indices]
            
            # Use balanced indices
            y_true_bal = y_true[all_balanced_indices]
            y_pred_bal = y_pred[all_balanced_indices]
            
            # Sanity check dimensions
            if len(y_true_bal) != len(y_pred_bal):
                print(f"Warning: Length mismatch for {exp_name} (True: {len(y_true_bal)}, Pred: {len(y_pred_bal)})")
                continue
                
            safe_name = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(common.FIGURE_DIR, f'cm_{safe_name}_n{target_n}_balanced.png')
            
            plot_confusion_matrix(y_true_bal, y_pred_bal, classes, f"Confusion Matrix (Balanced Test): {exp_name} (n={target_n})", save_path)
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

if __name__ == "__main__":
    main()

