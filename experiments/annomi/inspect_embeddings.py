import numpy as np
import sys

def inspect(path):
    print(f"Inspecting {path}...")
    try:
        data = np.load(path)
        for key in data.files:
            arr = data[key]
            print(f"--- {key} ---")
            print(f"  Shape: {arr.shape}")
            print(f"  Dtype: {arr.dtype}")
            print(f"  Min: {np.min(arr)}")
            print(f"  Max: {np.max(arr)}")
            print(f"  Mean: {np.mean(arr)}")
            print(f"  Std: {np.std(arr)}")
            print(f"  NaNs: {np.isnan(arr).sum()}")
            print(f"  Infs: {np.isinf(arr).sum()}")
            print(f"  Zeros: {(arr == 0).sum()} / {arr.size} ({100*(arr == 0).sum()/arr.size:.2f}%)")
            print(f"  First 5 rows:\n{arr[:5]}")
    except Exception as e:
        print(f"Error loading {path}: {e}")

inspect('annoMI/data/embeddings.npz')
inspect('annoMI/data/embeddings_prompted.npz')



