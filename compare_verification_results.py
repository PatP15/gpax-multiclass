import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import os

def compare_csvs(jax_path, torch_path, metric_col, join_cols, label):
    print(f"\n--- Comparing {label} ---")
    if not os.path.exists(jax_path) or not os.path.exists(torch_path):
        print(f"Skipping {label}: Files not found ({jax_path}, {torch_path})")
        return

    df_jax = pd.read_csv(jax_path)
    df_torch = pd.read_csv(torch_path)
    
    # Rename metric column to distinguish
    df_jax = df_jax.rename(columns={metric_col: f'{metric_col}_jax'})
    df_torch = df_torch.rename(columns={metric_col: f'{metric_col}_torch'})
    
    # Merge
    merged = pd.merge(df_jax, df_torch, on=join_cols, how='inner')
    
    print(f"Matched {len(merged)} records out of (JAX: {len(df_jax)}, Torch: {len(df_torch)})")
    
    if len(merged) == 0:
        print("No matched records found. Check join columns.")
        return

    # Compare
    diff = merged[f'{metric_col}_jax'] - merged[f'{metric_col}_torch']
    abs_diff = diff.abs()
    
    print(f"Mean Abs Diff: {abs_diff.mean():.6f}")
    print(f"Max Abs Diff:  {abs_diff.max():.6f}")
    print(f"99th Percentile Diff: {np.percentile(abs_diff, 99):.6f}")
    
    # Check for significant discrepancies
    bad = merged[abs_diff > 1e-3]
    if not bad.empty:
        print(f"Found {len(bad)} records with diff > 1e-3:")
        print(bad[join_cols + [f'{metric_col}_jax', f'{metric_col}_torch']].head())
    else:
        print("All differences <= 1e-3. Excellent parity.")

    # Plot Scatter
    plt.figure(figsize=(6, 6))
    plt.scatter(merged[f'{metric_col}_jax'], merged[f'{metric_col}_torch'], alpha=0.5)
    min_val = min(merged[f'{metric_col}_jax'].min(), merged[f'{metric_col}_torch'].min())
    max_val = max(merged[f'{metric_col}_jax'].max(), merged[f'{metric_col}_torch'].max())
    plt.plot([min_val, max_val], [min_val, max_val], 'k--')
    plt.xlabel(f'JAX {metric_col}')
    plt.ylabel(f'Torch {metric_col}')
    plt.title(f'{label}: JAX vs Torch Parity')
    plt.tight_layout()
    plt.savefig(f'parity_scatter_{label}.png')
    print(f"Saved parity_scatter_{label}.png")

def main():
    # 1. Compare Binary AUROC
    compare_csvs(
        'results_jax_auroc_binary.csv', 
        'results_torch_auroc_binary.csv',
        'AUROC',
        ['model', 'n_obs', 'method', 'task_type'],
        'Binary_AUROC'
    )

    # 2. Compare Multiclass AUROC
    compare_csvs(
        'results_jax_auroc_multi.csv', 
        'results_torch_auroc_multi.csv',
        'AUROC',
        ['model', 'n_obs', 'method', 'task_type'],
        'Multi_AUROC'
    )
    
    # 3. Compare Binary Simulation Results (Judged Probability)
    # Simulation results might not match exactly row-by-row because of random seeding differences 
    # unless we controlled seed perfectly in data generation.
    # In `simulate_ambiguous_data`, we used `np.random.uniform` for alphas. 
    # If we re-ran the script, the alphas would be different unless global seed is set.
    # Both scripts set random seeds inside loops, but the initial data generation might differ 
    # if `simulate_ambiguous_data` calls differ.
    # Actually, `simulate_ambiguous_data` uses `np.random.uniform`. 
    # If we run JAX script then Torch script, they are separate runs.
    # The seeds are reset? `main` doesn't set a global seed before calling simulate.
    # So exact row-by-row comparison of simulation might fail if data is random.
    # However, we can compare distributions or stats.
    
    # Let's skip direct row comparison for Sim/Unc unless we are sure they generated same data.
    # The `run_experiment` loop sets `seed = 42 + n_obs * 100 + r`.
    # `train_test_split` uses `random_state=42` or `seed`.
    # So for `run_experiment` (real data), the splits should be identical if sklearn behaves same.
    # The `simulate_ambiguous_data` function has `alphas = np.random.uniform(...)` with NO seed set inside the function.
    # So alphas will differ between runs.
    
    print("\nNote: Simulation and Uncertainty CSVs (Unc/Sim) contain random samples or generated data")
    print("that may not align row-by-row across separate process runs unless global seeds were identical.")
    print("Skipping direct row-wise comparison for Sim/Unc dataframes.")
    
if __name__ == "__main__":
    main()

