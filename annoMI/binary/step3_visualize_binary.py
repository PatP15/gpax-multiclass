import os
import sys
import numpy as np
import pandas as pd
import matplotlib
# Use Agg backend to avoid display errors on cluster
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

# Add project root to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
# Import binary common
sys.path.append(os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "binary"))
import annomi_common_binary as common

def plot_accuracy(df, save_path):
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='accuracy', hue='Scenario', marker='o')
    plt.title("Binary Classification Accuracy vs Sample Size")
    plt.ylabel("Accuracy")
    plt.xlabel("Number of Training Samples")
    plt.ylim(0.4, 1.0)
    plt.legend(title='Method', loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved accuracy plot to {save_path}")

def plot_auroc(df, save_path):
    # Filter for methods that support AUROC (Misclassification Detection)
    # LPE might not if n is too small or if it failed to calc
    if df.empty: return
    
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=df, x='n', y='auroc', hue='Scenario', marker='o')
    plt.title("Misclassification Detection (AUROC) - Binary Task")
    plt.ylabel("AUROC")
    plt.xlabel("Number of Training Samples")
    plt.ylim(0.4, 1.0)
    plt.legend(title='Method', loc='best')
    plt.grid(True, linestyle='--', alpha=0.3)
    plt.savefig(save_path, dpi=300)
    plt.close()
    print(f"Saved AUROC plot to {save_path}")

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
    print(f"Saved CM to {save_path}")

def balance_test_set_indices(y_true, seed=42):
    """
    Returns indices to balance the test set (undersample majority).
    """
    unique_classes, counts = np.unique(y_true, return_counts=True)
    min_count = np.min(counts)
    
    indices = []
    rng = np.random.default_rng(seed)
    
    for cls in unique_classes:
        cls_indices = np.where(y_true == cls)[0]
        if len(cls_indices) > min_count:
            selected = rng.choice(cls_indices, size=min_count, replace=False)
        else:
            selected = cls_indices
        indices.append(selected)
        
    return np.concatenate(indices)

def main():
    print("=== Step 3 Binary: Visualization ===")
    
    # 1. Load Results CSV
    results_path = os.path.join(common.RESULTS_DIR, 'results_binary_all.csv')
    if not os.path.exists(results_path):
        print(f"Results file not found: {results_path}")
        return
        
    df_res = pd.read_csv(results_path)
    # Update Scenario names for plot: (Prompted) -> (Expert)
    df_res['Scenario'] = df_res['Scenario'].replace({
        'Context (Prompted)': 'Context (Expert)',
        'Context + Quality (Prompted)': 'Context + Quality (Expert)',
        'Context (Few-Shot)': 'Context (Expert Few-Shot)'
    })
    
    # Plot Accuracy
    plot_accuracy(df_res, os.path.join(common.FIGURE_DIR, 'accuracy_comparison_binary.png'))
    
    # Plot AUROC
    plot_auroc(df_res, os.path.join(common.FIGURE_DIR, 'auroc_binary.png'))
    
    # 2. Confusion Matrices (Balanced Test Set)
    raw_path = os.path.join(common.DATA_DIR, 'raw_binary_all.npz')
    if not os.path.exists(raw_path):
        print(f"Raw data file not found: {raw_path}")
        return
        
    raw_data = np.load(raw_path)
    
    # Scenarios to plot (match keys in raw_binary_all.npz)
    scenarios = {
        'Context': 'Context',
        'Context_Qual': 'Context + Quality',
        'Prompted': 'Context (Expert)',
        'Prompted_Qual': 'Context + Quality (Expert)',
        'FewShot': 'Context (Expert Few-Shot)'
    }
    classes = ['Non-Change', 'Change']
    
    # Use max sample size
    max_n = common.SAMPLE_SIZES[-1]
    
    for key_prefix, title_name in scenarios.items():
        key_true = f'{key_prefix}_n{max_n}_y_true'
        key_pred = f'{key_prefix}_n{max_n}_y_pred'
        
        if key_true not in raw_data or key_pred not in raw_data:
            print(f"Missing data for {title_name} ({key_prefix}) at n={max_n}")
            continue
            
        y_true = raw_data[key_true]
        y_pred = raw_data[key_pred]
        
        # Balance Test Set for Visualization
        bal_idx = balance_test_set_indices(y_true)
        y_true_bal = y_true[bal_idx]
        y_pred_bal = y_pred[bal_idx]
        
        save_path = os.path.join(common.FIGURE_DIR, f'cm_{key_prefix.lower()}_n{max_n}_binary_balanced.png')
        plot_confusion_matrix(
            y_true_bal, y_pred_bal, classes, 
            f"Binary CM: {title_name} (n={max_n}, Balanced Test)", 
            save_path
        )

if __name__ == "__main__":
    main()
