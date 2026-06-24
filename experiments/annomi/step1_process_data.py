import sys
print("DEBUG: Step 1 Script Starting...", flush=True)

import os
# Force JAX to use CPU to avoid fighting for GPU memory with PyTorch
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
print("DEBUG: Imported numpy", flush=True)
import pandas as pd
print("DEBUG: Imported pandas", flush=True)
from sklearn.decomposition import PCA
print("DEBUG: Imported sklearn", flush=True)
import annomi_common as common
print("DEBUG: Imported annomi_common", flush=True)
import gc
import torch
print("DEBUG: Imported torch", flush=True)

# Unprompted Instruction
INSTRUCTION = "choose one of the three options (change, sustain, or neutral)"

def main():
    print("=== Step 1: Process Data & Extract Embeddings (Unprompted Split) ===")
    
    # 1. Load Data (Train/Test Split)
    df_train, df_test = common.load_annomi_data()
    
    # Save processed dataframes
    df_train.to_csv(os.path.join(common.DATA_DIR, "annomi_train_processed.csv"), index=False)
    df_test.to_csv(os.path.join(common.DATA_DIR, "annomi_test_processed.csv"), index=False)
    
    # Labels
    y_train_mot = df_train['y_mot'].values
    y_train_qual = df_train['y_qual'].values
    y_test_mot = df_test['y_mot'].values
    y_test_qual = df_test['y_qual'].values

    # Load Model using common logic
    print(f"Loading Model {common.MODEL_NAME}...")
    model, tokenizer = common.load_model()
    print("Model loaded successfully.")
    
    # 2. Extract Embeddings (Helper Function)
    def process_split(df, prefix):
        print(f"\n--- Processing {prefix} Set ---")
        X_client = df['client_text'].tolist()
        X_therapist = df['therapist_text'].tolist()
        y_qual = df['y_qual'].values
        
        # 1. Context (Standard)
        # Format: [Inst] INPUT: Therapist: ... Client: ... OUTPUT:
        X_context = [f"{INSTRUCTION}\nINPUT: Therapist: {t}\nClient: {c}\nOUTPUT:" for t, c in zip(X_therapist, X_client)]
        
        # 2. Context + Quality
        # Format: [Inst] INPUT: Therapist: ... (Quality: ...) Client: ... OUTPUT:
        qual_map = {0: 'Low', 1: 'High'}
        X_context_qual = []
        for t, c, q in zip(X_therapist, X_client, y_qual):
            q_str = qual_map.get(q, 'Low') # Default to Low if unknown
            X_context_qual.append(f"{INSTRUCTION}\nINPUT: Therapist: {t} (Quality: {q_str})\nClient: {c}\nOUTPUT:")
        
        # Embed
        print(f"Embedding Context ({len(X_context)})...")
        emb_context = common.get_embeddings(X_context, model=model, tokenizer=tokenizer, batch_size=1, max_length=8192)
        gc.collect(); torch.cuda.empty_cache()
        
        print(f"Embedding Context+Quality ({len(X_context_qual)})...")
        emb_context_qual = common.get_embeddings(X_context_qual, model=model, tokenizer=tokenizer, batch_size=1, max_length=8192)
        gc.collect(); torch.cuda.empty_cache()
        
        return emb_context, emb_context_qual

    # Extract
    emb_train_ctx, emb_train_ctx_q = process_split(df_train, "Train")
    emb_test_ctx, emb_test_ctx_q = process_split(df_test, "Test")

    # Cleanup Model
    del model
    del tokenizer
    if 'pipe' in locals(): del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # 3. PCA
    print("\n--- Reducing Dimensions (PCA -> 64) ---")
    
    def apply_pca_split(train_emb, test_emb):
        # Check for NaNs/Infs
        if np.any(np.isnan(train_emb)) or np.any(np.isinf(train_emb)):
            print("Warning: NaNs or Infs found in Train Embeddings. Replacing with 0.")
            train_emb = np.nan_to_num(train_emb)
            
        if np.any(np.isnan(test_emb)) or np.any(np.isinf(test_emb)):
            print("Warning: NaNs or Infs found in Test Embeddings. Replacing with 0.")
            test_emb = np.nan_to_num(test_emb)
            
        pca = PCA(n_components=64)
        train_pca = pca.fit_transform(train_emb)
        test_pca = pca.transform(test_emb)
        return train_pca, test_pca
        
    X_train_context, X_test_context = apply_pca_split(emb_train_ctx, emb_test_ctx)
    X_train_context_qual, X_test_context_qual = apply_pca_split(emb_train_ctx_q, emb_test_ctx_q)
    
    # 4. Save
    save_path = os.path.join(common.DATA_DIR, 'embeddings.npz')
    np.savez(
        save_path,
        # Train
        X_train_context=X_train_context,
        X_train_context_qual=X_train_context_qual,
        y_train_mot=y_train_mot,
        y_train_qual=y_train_qual,
        # Test
        X_test_context=X_test_context,
        X_test_context_qual=X_test_context_qual,
        y_test_mot=y_test_mot,
        y_test_qual=y_test_qual
    )
    print(f"Saved embeddings to {save_path}")

if __name__ == "__main__":
    main()
