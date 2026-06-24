import annomi_common as common
import numpy as np
import pandas as pd

def main():
    print("Loading data...")
    # This now loads the new split
    df_train, df_test = common.load_annomi_data()
    
    print("\n--- Training Set Analysis ---")
    print(f"Total samples: {len(df_train)}")
    print("Class distribution (Motivation):")
    print(df_train['y_mot'].value_counts().sort_index())
    
    # Calculate max balanced size
    min_class_train = df_train['y_mot'].value_counts().min()
    print(f"\nMinimum class count in Train: {min_class_train}")
    print(f"Max balanced dataset size would be: {min_class_train * 3}")
    
    print("\n--- Test Set Analysis ---")
    print(f"Total samples: {len(df_test)}")
    print("Class distribution (Motivation):")
    print(df_test['y_mot'].value_counts().sort_index())

if __name__ == "__main__":
    main()

