import os
import numpy as np
import pandas as pd
import annomi_common as common

def main():
    print("=== Step 2: Experiment 4 (Context + Quality - Prompted) ===")
    
    # Load Prompted Data
    data = np.load(os.path.join(common.DATA_DIR, 'embeddings_prompted.npz'))
    X_train = data['X_train_context_qual']
    y_train = data['y_train_mot']
    X_test = data['X_test_context_qual']
    y_test = data['y_test_mot']
    
    # Run Training Loop
    res, raw_data = common.run_training_loop(
        X_train, y_train, X_test, y_test,
        common.SAMPLE_SIZES, 'multiclass', return_raw=True
    )
    res['Scenario'] = 'Context + Quality (Prompted)'
    
    # Save Results
    save_path = os.path.join(common.RESULTS_DIR, 'results_exp4_prompted_quality.csv')
    res.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

    # Save Raw Data
    raw_path = os.path.join(common.DATA_DIR, 'raw_exp4_prompted_quality.npz')
    save_dict = {}
    for n, d in raw_data.items():
        for k, v in d.items():
            save_dict[f'n{n}_{k}'] = v
    np.savez(raw_path, **save_dict)
    print(f"Saved raw data to {raw_path}")

if __name__ == "__main__":
    main()




