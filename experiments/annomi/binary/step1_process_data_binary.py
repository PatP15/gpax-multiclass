import sys
print("DEBUG: Step 1 Binary Script Starting...", flush=True)

import os
# Force JAX to use CPU to avoid fighting for GPU memory with PyTorch
os.environ['JAX_PLATFORMS'] = 'cpu'

import numpy as np
print("DEBUG: Imported numpy", flush=True)
import pandas as pd
print("DEBUG: Imported pandas", flush=True)
from sklearn.decomposition import PCA
print("DEBUG: Imported sklearn", flush=True)
import annomi_common_binary as common
print("DEBUG: Imported annomi_common_binary", flush=True)
import gc
import torch
print("DEBUG: Imported torch", flush=True)

INSTRUCTION = "Classify the client's motivation. Respond with only one word (Change or Non-Change) and nothing else."

# Define the Prompts
MOTIVATION_PROMPT = """You are an expert psychotherapist and supervisor for Motivational Interviewing (MI) sessions. Your task is to analyze the following dialogue between a therapist and a client.

Specifically, you must determine the **client's current motivation toward change** based on their utterance.
Classify the client's speech into one of two categories:
1. **Change Talk**: The client expresses a desire, ability, reason, or need to change their behavior.
2. **Non-Change Talk**: The client expresses a desire to maintain the status quo, argues against change (Sustain Talk), or provides a Neutral response.

Analyze the context provided by the therapist's previous statement and the client's response.

INPUT: Therapist: {therapist_text}
Client: {client_text}
Respond with only one word (Change or Non-Change) and nothing else.
OUTPUT:"""

MOTIVATION_PROMPT_QUAL = """You are an expert psychotherapist and supervisor for Motivational Interviewing (MI) sessions. Your task is to analyze the following dialogue between a therapist and a client.

Specifically, you must determine the **client's current motivation toward change** based on their utterance.
Classify the client's speech into one of two categories:
1. **Change Talk**: The client expresses a desire, ability, reason, or need to change their behavior.
2. **Non-Change Talk**: The client expresses a desire to maintain the status quo, argues against change (Sustain Talk), or provides a Neutral response.

Analyze the context provided by the therapist's previous statement (annotated with Quality) and the client's response.

INPUT: Therapist: {therapist_text} (Quality: {quality})
Client: {client_text}
Respond with only one word (Change or Non-Change) and nothing else.
OUTPUT:"""

# Few-Shot Examples (2 per class for Change, Non-Change)
FEW_SHOT_EXAMPLES = """
Below are some examples.

Example 1:
INPUT: Therapist: What do you think you might do?
Client: I really want to stop smoking for my health.
OUTPUT: Change

Example 2:
INPUT: Therapist: How important is this to you?
Client: I could probably cut down if I really tried.
OUTPUT: Change

Example 3:
INPUT: Therapist: Why don't you want to change?
Client: I don't think I have a problem, honestly.
OUTPUT: Non-Change

Example 4:
INPUT: Therapist: Good morning, how are you?
Client: I'm doing okay, thanks.
OUTPUT: Non-Change
"""

MOTIVATION_PROMPT_FEWSHOT = """You are an expert psychotherapist and supervisor for Motivational Interviewing (MI) sessions. Your task is to analyze the following dialogue between a therapist and a client.

Specifically, you must determine the **client's current motivation toward change** based on their utterance.
Classify the client's speech into one of two categories:
1. **Change Talk**: The client expresses a desire, ability, reason, or need to change their behavior.
2. **Non-Change Talk**: The client expresses a desire to maintain the status quo, argues against change (Sustain Talk), or provides a Neutral response.

Analyze the context provided by the therapist's previous statement and the client's response.
{few_shot_examples}

Given the following input, output the class (Change Talk or Non-Change Talk).
INPUT: Therapist: {therapist_text}
Client: {client_text}
Respond with only one word (Change or Non-Change) and nothing else.
OUTPUT:"""

def main():
    print("=== Step 1 Binary: Process Data & Extract Embeddings (Split) ===")
    
    # 1. Load Data (Train/Test Split) - Already does Binary Mapping in common
    df_train, df_test = common.load_annomi_data()
    
    # Save processed dataframes
    df_train.to_csv(os.path.join(common.DATA_DIR, "annomi_train_processed_binary.csv"), index=False)
    df_test.to_csv(os.path.join(common.DATA_DIR, "annomi_test_processed_binary.csv"), index=False)
    
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
        qual_map = {0: 'Low', 1: 'High'}
        
        # 1. Context (Unprompted)
        X_context = [f"{INSTRUCTION}\nINPUT: Therapist: {t}\nClient: {c}\nOUTPUT:" for t, c in zip(X_therapist, X_client)]
        
        # 2. Context + Quality (Unprompted)
        X_context_qual = []
        for t, c, q in zip(X_therapist, X_client, y_qual):
            q_str = qual_map.get(q, 'Low')
            X_context_qual.append(f"{INSTRUCTION}\nINPUT: Therapist: {t} (Quality: {q_str})\nClient: {c}\nOUTPUT:")
            
        # 3. Context (Prompted)
        X_context_prompted = []
        for t, c in zip(X_therapist, X_client):
            prompted_text = MOTIVATION_PROMPT.format(therapist_text=t, client_text=c)
            X_context_prompted.append(prompted_text)
            
        # 4. Context + Quality (Prompted)
        X_context_qual_prompted = []
        for t, c, q in zip(X_therapist, X_client, y_qual):
            q_str = qual_map.get(q, 'Low')
            prompted_text = MOTIVATION_PROMPT_QUAL.format(therapist_text=t, client_text=c, quality=q_str)
            X_context_qual_prompted.append(prompted_text)

        # 5. Context + Few-Shot (Prompted)
        X_context_fewshot = []
        for t, c in zip(X_therapist, X_client):
            prompted_text = MOTIVATION_PROMPT_FEWSHOT.format(
                few_shot_examples=FEW_SHOT_EXAMPLES,
                therapist_text=t, 
                client_text=c
            )
            X_context_fewshot.append(prompted_text)
        
        # Embed
        print(f"Embedding Context ({len(X_context)})...")
        emb_context = common.get_embeddings(X_context, model=model, tokenizer=tokenizer, batch_size=1, max_length=512)
        gc.collect(); torch.cuda.empty_cache()
        
        print(f"Embedding Context+Quality ({len(X_context_qual)})...")
        emb_context_qual = common.get_embeddings(X_context_qual, model=model, tokenizer=tokenizer, batch_size=1, max_length=512)
        gc.collect(); torch.cuda.empty_cache()
        
        print(f"Embedding Prompted ({len(X_context_prompted)})...")
        emb_context_p = common.get_embeddings(X_context_prompted, model=model, tokenizer=tokenizer, batch_size=1, max_length=512)
        gc.collect(); torch.cuda.empty_cache()
        
        print(f"Embedding Prompted+Quality ({len(X_context_qual_prompted)})...")
        emb_context_qual_p = common.get_embeddings(X_context_qual_prompted, model=model, tokenizer=tokenizer, batch_size=1, max_length=512)
        gc.collect(); torch.cuda.empty_cache()
        
        print(f"Embedding Prompted+FewShot ({len(X_context_fewshot)})...")
        # Increase max_length for few-shot prompt
        emb_context_fs = common.get_embeddings(X_context_fewshot, model=model, tokenizer=tokenizer, batch_size=1, max_length=1024)
        gc.collect(); torch.cuda.empty_cache()
        
        return emb_context, emb_context_qual, emb_context_p, emb_context_qual_p, emb_context_fs

    # Extract
    emb_train_ctx, emb_train_cq, emb_train_p, emb_train_qp, emb_train_fs = process_split(df_train, "Train")
    emb_test_ctx, emb_test_cq, emb_test_p, emb_test_qp, emb_test_fs = process_split(df_test, "Test")

    # Cleanup Model
    del model
    del tokenizer
    if 'pipe' in locals(): del pipe
    torch.cuda.empty_cache()
    gc.collect()

    # 3. PCA
    print("\n--- Reducing Dimensions (PCA -> 64) ---")
    
    def apply_pca_split(train_emb, test_emb):
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
        
    X_train_ctx, X_test_ctx = apply_pca_split(emb_train_ctx, emb_test_ctx)
    X_train_cq, X_test_cq = apply_pca_split(emb_train_cq, emb_test_cq)
    X_train_p, X_test_p = apply_pca_split(emb_train_p, emb_test_p)
    X_train_qp, X_test_qp = apply_pca_split(emb_train_qp, emb_test_qp)
    X_train_fs, X_test_fs = apply_pca_split(emb_train_fs, emb_test_fs)
    
    # 4. Save
    save_path = os.path.join(common.DATA_DIR, 'embeddings_binary.npz')
    np.savez(
        save_path,
        # Train
        X_train_context=X_train_ctx,
        X_train_context_qual=X_train_cq,
        X_train_prompted=X_train_p,
        X_train_prompted_qual=X_train_qp,
        X_train_fewshot=X_train_fs,
        y_train_mot=y_train_mot,
        y_train_qual=y_train_qual,
        # Test
        X_test_context=X_test_ctx,
        X_test_context_qual=X_test_cq,
        X_test_prompted=X_test_p,
        X_test_prompted_qual=X_test_qp,
        X_test_fewshot=X_test_fs,
        y_test_mot=y_test_mot,
        y_test_qual=y_test_qual
    )
    print(f"Saved binary embeddings to {save_path}")

if __name__ == "__main__":
    main()
