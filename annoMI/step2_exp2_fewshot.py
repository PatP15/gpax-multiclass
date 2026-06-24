import os
import numpy as np
import pandas as pd
import annomi_common as common

def main():
    print("=== Step 2: Experiment 2 (Context + Few-Shot) ===")
    
    # Load Prompted Data
    data = np.load(os.path.join(common.DATA_DIR, 'embeddings_prompted.npz'))
    X_train = data['X_train_fewshot']
    y_train = data['y_train_mot']
    X_test = data['X_test_fewshot']
    y_test = data['y_test_mot']
    
    # Run Training Loop
    res, raw_data = common.run_training_loop(
        X_train, y_train, X_test, y_test,
        common.SAMPLE_SIZES, 'multiclass', return_raw=True
    )
    res['Scenario'] = 'Context (Few-Shot)'
    
    # Save Results
    save_path = os.path.join(common.RESULTS_DIR, 'results_exp2_fewshot.csv')
    res.to_csv(save_path, index=False)
    print(f"Saved results to {save_path}")

    # Save Raw Data
    raw_path = os.path.join(common.DATA_DIR, 'raw_exp2_fewshot.npz')
    save_dict = {}
    for n, d in raw_data.items():
        for k, v in d.items():
            save_dict[f'n{n}_{k}'] = v
    np.savez(raw_path, **save_dict)
    print(f"Saved raw data to {raw_path}")

if __name__ == "__main__":
    main()


