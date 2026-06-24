import os
import numpy as np
import jax
import jax.numpy as jnp
import cnn_model
import gpp_common as common

def verify():
    print("=== Verifying Step 1 Outputs ===")
    
    # 1. Check data_labels.npz
    print("\n1. Checking data_labels.npz...")
    try:
        data = np.load(os.path.join(common.EMBEDDING_DIR, 'data_labels.npz'), allow_pickle=True)
        print("Keys:", data.files)
        
        for key in ['train_indices', 'val_indices', 'test_indices', 'images', 'labels_raw']:
            if key in data:
                print(f"  {key}: {data[key].shape}")
            else:
                print(f"  MISSING: {key}")
                
        n_train = len(data['train_indices'])
        n_val = len(data['val_indices'])
        n_test = len(data['test_indices'])
        n_total = len(data['images'])
        
        print(f"  Split counts: Train={n_train}, Val={n_val}, Test={n_test}, Total={n_total}")
        if n_train + n_val + n_test != n_total:
            print("  WARNING: Splits do not sum to total images (indices might be subset of full if subsetting was used differently?)")
        else:
            print("  Splits sum correctly.")
            
    except Exception as e:
        print(f"FAILED to load data_labels.npz: {e}")
        return

    # 2. Check Embeddings
    print("\n2. Checking Embeddings...")
    for m in ['M1', 'M2', 'M3']:
        path = os.path.join(common.EMBEDDING_DIR, f'embeddings_{m}.npy')
        if os.path.exists(path):
            emb = np.load(path)
            print(f"  {m} Embeddings: {emb.shape}")
            if emb.shape[0] != n_total:
                print(f"  WARNING: Embedding count {emb.shape[0]} != Image count {n_total}")
        else:
            print(f"  MISSING: {path}")

    # 3. Check Checkpoints & Inference
    print("\n3. Checking Checkpoints & Inference...")
    
    # Initialize a dummy state to load into
    rng = jax.random.PRNGKey(0)
    # Need num_classes to init
    # M1: 64, M2: 8, M3: 8
    model_specs = [('M1', 64), ('M2', 8), ('M3', 8)]
    
    dummy_input = jnp.ones((1, 64, 64, 3), dtype=jnp.float32)
    
    for m, k in model_specs:
        print(f"  Testing {m} (classes={k})...")
        try:
            # Create fresh state
            state = cnn_model.create_train_state(rng, num_classes=k)
            
            # Load
            state_loaded = common.load_checkpoint(state, common.EMBEDDING_DIR, m)
            
            if state_loaded is state:
                print(f"    WARNING: {m} checkpoint not found or not loaded (state object identical).")
                # verify dict content if possible or check step
            else:
                print(f"    Loaded checkpoint step: {int(state_loaded.step)}")
                
            # Inference
            logits, emb = state_loaded.apply_fn({'params': state_loaded.params}, dummy_input, return_embeddings=True)
            print(f"    Inference successful. Logits: {logits.shape}, Emb: {emb.shape}")
            
        except Exception as e:
            print(f"    FAILED {m}: {e}")

    print("\n=== Verification Complete ===")

if __name__ == "__main__":
    verify()

