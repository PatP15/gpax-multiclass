import h5py
import numpy as np

DATA_PATH = '3dshapes.h5'

def check_data():
    try:
        f = h5py.File(DATA_PATH, 'r')
        labels = f['labels'][:1000] # Read first 1000
        f.close()
        
        print("Labels Shape:", labels.shape)
        for i in range(6):
            vals = labels[:, i]
            uniques = np.unique(vals)
            print(f"Col {i}: {len(uniques)} unique values. Min: {np.min(vals)}, Max: {np.max(vals)}")
            if len(uniques) < 20:
                print(f"  Values: {uniques}")
                
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    check_data()




