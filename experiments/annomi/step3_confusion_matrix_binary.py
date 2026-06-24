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
        'Cascading': 'raw_exp4.npz', 
        'LPE': 'raw_lpe.npz'
    }
    
    # Original Mapping: 'sustain': 0, 'neutral': 1, 'change': 2
    # New Binary Mapping: {0: 0, 1: 0, 2: 1}  (Sustain/Neutral -> Non-Change (0), Change -> Change (1))
    binary_classes = ['Non-Change', 'Change']
    
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
                    n_str = key.split('_')[0][1:] 
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

            # --- Convert to Binary (Lump Sustain & Neutral) ---
            # 0=Sustain, 1=Neutral, 2=Change
            # Map 0,1 -> 0 (Non-Change)
            # Map 2 -> 1 (Change)
            y_true_bin = np.where(y_true == 2, 1, 0)
            y_pred_bin = np.where(y_pred == 2, 1, 0)

            # --- Balance the Test Set for Visualization (Binary) ---
            unique_classes, counts = np.unique(y_true_bin, return_counts=True)
            min_count = np.min(counts)
            print(f"  Balancing test set (Binary) for CM: Reducing classes to {min_count} samples.")
            
            balanced_indices = []
            rng = np.random.default_rng(42)
            
            for cls in unique_classes:
                cls_indices = np.where(y_true_bin == cls)[0]
                if len(cls_indices) > min_count:
                    selected = rng.choice(cls_indices, size=min_count, replace=False)
                else:
                    selected = cls_indices
                balanced_indices.append(selected)
                
            all_balanced_indices = np.concatenate(balanced_indices)
            
            y_true_bal = y_true_bin[all_balanced_indices]
            y_pred_bal = y_pred_bin[all_balanced_indices]
            
            # Calculate Binary Accuracy on Balanced Set
            acc = np.mean(y_true_bal == y_pred_bal)
            print(f"  Binary Accuracy (Balanced): {acc:.4f}")

            safe_name = exp_name.lower().replace(" ", "_").replace("(", "").replace(")", "")
            save_path = os.path.join(common.FIGURE_DIR, f'cm_{safe_name}_n{target_n}_binary_balanced.png')
            
            plot_confusion_matrix(y_true_bal, y_pred_bal, binary_classes, f"Binary CM (Sustain+Neutral vs Change): {exp_name}", save_path)
            
        except Exception as e:
            print(f"Error processing {exp_name}: {e}")

if __name__ == "__main__":
    main()


