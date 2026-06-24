import numpy as np
import pandas as pd
import os

# Standalone script to avoid complex dependencies
def find_examples():
    data_dir = "annoMI/data/gemma"
    raw_path = os.path.join(data_dir, 'raw_exp2_fewshot.npz')
    
    # Need to load the *original* dataframe to get text
    # But wait, step1_process saves 'annomi_test_processed.csv'.
    # Let's check if it exists.
    test_csv_path = os.path.join(data_dir, 'annomi_test_processed.csv')
    
    if not os.path.exists(raw_path):
        print(f"Raw data not found: {raw_path}")
        return
    if not os.path.exists(test_csv_path):
        print(f"Test CSV not found: {test_csv_path}")
        return
        
    print(f"Loading {raw_path}...")
    data = np.load(raw_path)
    
    print(f"Loading {test_csv_path}...")
    df_test = pd.read_csv(test_csv_path)
    
    # Use max sample size
    n = 2400
    key_prefix = f"n{n}_"
    
    if f'{key_prefix}epistemic' not in data:
        print(f"Key {key_prefix}epistemic not found.")
        print(f"Keys available: {list(data.keys())}")
        return
        
    epis = data[f'{key_prefix}epistemic']
    alea = data[f'{key_prefix}aleatoric']
    preds = data[f'{key_prefix}y_pred']
    y_true = data[f'{key_prefix}y_true']
    
    # Handle scalar/array issues if any
    if epis.ndim > 1: epis = epis.flatten()
    if alea.ndim > 1: alea = alea.flatten()
    
    # Calculate Thresholds
    e_low = np.percentile(epis, 25)
    e_high = np.percentile(epis, 75)
    a_low = np.percentile(alea, 25)
    a_high = np.percentile(alea, 75)
    
    print(f"Epistemic (Certainty): Low(25%) < {e_low:.2f}, High(75%) > {e_high:.2f}")
    print(f"Aleatoric (Ambiguity): Low(25%) < {a_low:.2f}, High(75%) > {a_high:.2f}")
    
    quadrants = {
        "High Alea / High Episteme (Certain Ambiguity)": (alea > a_high) & (epis > e_high),
        "High Alea / Low Episteme (Total Confusion)": (alea > a_high) & (epis < e_low),
        "Low Alea / High Episteme (Confident Clarity)": (alea < a_low) & (epis > e_high),
        "Low Alea / Low Episteme (Cautious Clarity)": (alea < a_low) & (epis < e_low)
    }
    
    label_map = {0: 'Sustain', 1: 'Neutral', 2: 'Change'}
    
    for name, mask in quadrants.items():
        indices = np.where(mask)[0]
        print(f"\n=== {name} (Count: {len(indices)}) ===")
        
        if len(indices) == 0:
            print("No examples found.")
            continue
            
        # Pick one random example
        idx = indices[0] # First one
        
        # Verify index is within bounds
        if idx >= len(df_test):
            print(f"Index {idx} out of bounds for DF (len {len(df_test)})")
            continue
            
        row = df_test.iloc[idx]
        print(f"Therapist: {row['therapist_text']}")
        print(f"Client: {row['client_text']}")
        print(f"GT: {label_map[row['y_mot']]}")
        print(f"Pred: {label_map[preds[idx]]}")
        print(f"Epistemic: {epis[idx]:.3f}")
        print(f"Aleatoric: {alea[idx]:.3f}")

if __name__ == "__main__":
    find_examples()
