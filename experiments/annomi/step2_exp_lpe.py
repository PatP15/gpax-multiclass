import os
import numpy as np
import pandas as pd
import annomi_common as common

def main():
    print("=== Step 2: LPE Baseline Experiments (Context + Few-Shot) ===")
    
    # Load Prompted Data (which now contains Few-Shot)
    # Ensure this matches where Step 1 Prompted saves data
    data = np.load(os.path.join(common.DATA_DIR, 'embeddings_prompted.npz'))
    X_train = data['X_train_fewshot']
    y_train = data['y_train_mot']
    X_test = data['X_test_fewshot']
    y_test = data['y_test_mot']
    
    # Run LPE Training Loop
    print("Running LPE on Context+FewShot embeddings...")
    res, raw_data = common.run_training_loop(
        X_train, y_train, X_test, y_test,
        common.SAMPLE_SIZES, 'multiclass', method='lpe', return_raw=True
    )
    res['Scenario'] = 'LPE (Expert Few-Shot)'
    
    # Save Results
    res_path = os.path.join(common.RESULTS_DIR, 'results_lpe.csv')
    res.to_csv(res_path, index=False)
    print(f"Saved results to {res_path}")
    
    # Save Raw Data
    raw_path = os.path.join(common.DATA_DIR, 'raw_lpe.npz')
    save_dict = {}
    for n, d in raw_data.items():
        for k, v in d.items():
            save_dict[f'n{n}_{k}'] = v
            
    np.savez(raw_path, **save_dict)
    print(f"Saved raw data to {raw_path}")

if __name__ == "__main__":
    main()
