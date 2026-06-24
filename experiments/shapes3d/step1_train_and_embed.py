import os
import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
import cnn_model
import gpp_common as common

def main():
    print("=== Step 1: Train Models & Generate Embeddings ===", flush=True)
    common.ensure_dirs()
    
    # Load Data
    # Use FULL dataset as verified
    images, labels_raw, labels_dict = common.load_data(subset_size=None)
    
    # Save labels for later steps
    save_dict = {k: v for k, v in labels_dict.items()}
    # images are too big to save in npz likely (480k * 64*64*3 * byte ~ 5GB). 
    # np.savez compresses? No.
    # We should probably NOT save the images in data_labels.npz if they are huge.
    # But step 2 needs them?
    # Step 2 loads embeddings. Does it load images?
    # step2 uses `data['images']`? No, it loads `data_labels.npz` and extracts keys.
    # Let's check step2.
    # Step 2 loads: `labels_dict = {k: data[k] ...}`
    # It does NOT use images.
    # But wait, `step1` currently saves `save_dict['images'] = images`.
    # If I try to save 480k images to .npz, it will be huge and slow.
    # The images are already in `3dshapes.h5`.
    # I should STOP saving images in `data_labels.npz`.
    
    # Remove images from save_dict to save space
    # save_dict['images'] = images 
    save_dict['labels_raw'] = labels_raw
    
    # Generate Train/Val/Test Split INDICES to be consistent
    N = len(labels_raw) # Use labels length since we might not have images loaded if we change logic, but here we do.
    indices = np.arange(N)
    # 60% Train, 20% Val, 20% Test
    train_idx, test_val_idx = train_test_split(indices, test_size=0.4, random_state=42)
    val_idx, test_idx = train_test_split(test_val_idx, test_size=0.5, random_state=42)
    
    save_dict['train_indices'] = train_idx
    save_dict['val_indices'] = val_idx
    save_dict['test_indices'] = test_idx
    
    np.savez(os.path.join(common.EMBEDDING_DIR, 'data_labels.npz'), **save_dict)
    print(f"Saved data indices (Train: {len(train_idx)}, Val: {len(val_idx)}, Test: {len(test_idx)}).", flush=True)
    
    # Models Configuration
    models_config = [('M1', 64), ('M2', 8), ('M3', 8)]
    
    for m, k in models_config:
        print(f"\nProcessing Model {m}...", flush=True)
        y = labels_dict[m]
        
        # Train using explicit indices
        X_train = images[train_idx]
        y_train = y[train_idx]
        X_val = images[val_idx]
        y_val = y[val_idx]
        
        print(f"Training {m} with {len(X_train)} samples...", flush=True)
        state = cnn_model.train_model(X_train, y_train, X_val, y_val, num_classes=k, num_epochs=1, batch_size=64, verbose=True)
        
        # Save Checkpoint
        print(f"Saving checkpoint for {m}...", flush=True)
        common.save_checkpoint(state, common.EMBEDDING_DIR, m)
        
        # Embed
        print(f"Generating Embeddings for {m}...", flush=True)
        X_emb = common.get_embeddings_batched(state, images)
        
        # Save
        save_path = os.path.join(common.EMBEDDING_DIR, f'embeddings_{m}.npy')
        np.save(save_path, X_emb)
        print(f"Saved embeddings to {save_path}")

    print("\nStep 1 Completed.")

if __name__ == "__main__":
    main()

