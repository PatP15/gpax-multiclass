import os
import numpy as np
import pandas as pd
import annomi_common_binary as common

def main():
    print("=== Step 2 Binary: Experiments (5 Variations) ===")
    
    # Load Data
    data = np.load(os.path.join(common.DATA_DIR, 'embeddings_binary.npz'))
    
    y_train = data['y_train_mot']
    y_test = data['y_test_mot']
    
    # --- Experiment 1: Context (Unprompted) ---
    print("\n--- Experiment 1: Context (Unprompted) ---")
    res1, raw1 = common.run_training_loop(
        data['X_train_context'], y_train, 
        data['X_test_context'], y_test, 
        common.SAMPLE_SIZES, 'binary', return_raw=True
    )
    res1['Scenario'] = 'Context'
    
    # --- Experiment 2: Context + Quality (Unprompted) ---
    print("\n--- Experiment 2: Context + Quality (Unprompted) ---")
    res2, raw2 = common.run_training_loop(
        data['X_train_context_qual'], y_train, 
        data['X_test_context_qual'], y_test, 
        common.SAMPLE_SIZES, 'binary', return_raw=True
    )
    res2['Scenario'] = 'Context + Quality'

    # --- Experiment 3: Context (Prompted) ---
    print("\n--- Experiment 3: Context (Prompted) ---")
    res3, raw3 = common.run_training_loop(
        data['X_train_prompted'], y_train, 
        data['X_test_prompted'], y_test, 
        common.SAMPLE_SIZES, 'binary', return_raw=True
    )
    res3['Scenario'] = 'Context (Prompted)'
    
    # --- Experiment 4: Context + Quality (Prompted) ---
    print("\n--- Experiment 4: Context + Quality (Prompted) ---")
    res4, raw4 = common.run_training_loop(
        data['X_train_prompted_qual'], y_train, 
        data['X_test_prompted_qual'], y_test, 
        common.SAMPLE_SIZES, 'binary', return_raw=True
    )
    res4['Scenario'] = 'Context + Quality (Prompted)'

    # --- Experiment 5: Context + Few-Shot (Prompted) ---
    print("\n--- Experiment 5: Context + Few-Shot (Prompted) ---")
    res5, raw5 = common.run_training_loop(
        data['X_train_fewshot'], y_train, 
        data['X_test_fewshot'], y_test, 
        common.SAMPLE_SIZES, 'binary', return_raw=True
    )
    res5['Scenario'] = 'Context (Few-Shot)'
    
    # --- Save All ---
    pd.concat([res1, res2, res3, res4, res5]).to_csv(os.path.join(common.RESULTS_DIR, 'results_binary_all.csv'), index=False)
    
    # Save Raw
    raw_path = os.path.join(common.DATA_DIR, 'raw_binary_all.npz')
    save_dict = {}
    for name, d in [('Context', raw1), ('Context_Qual', raw2), ('Prompted', raw3), ('Prompted_Qual', raw4), ('FewShot', raw5)]:
        for n, sub_d in d.items():
            for k, v in sub_d.items():
                save_dict[f'{name}_n{n}_{k}'] = v
    np.savez(raw_path, **save_dict)
    print(f"Saved all results to {common.RESULTS_DIR}")

if __name__ == "__main__":
    main()
