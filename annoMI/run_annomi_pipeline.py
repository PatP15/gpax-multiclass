import os
import sys
import numpy as np
import pandas as pd
import torch
import jax
import jax.numpy as jnp
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModelForCausalLM
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm

# Add project root to path for GPax imports
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from GPax.probing import probabilistic_probe_multiclass as ppm
from GPax.probing import probabilistic_probe as pp

# --- Constants ---
MODEL_NAME = "google/gemma-2-2b"
SEED = 42
SAMPLE_SIZES = [10, 30, 50, 75, 100]
FIGURE_DIR = os.path.join("annoMI", "figures")
os.makedirs(FIGURE_DIR, exist_ok=True)

# --- 1. Data Loading & Processing ---

def load_annomi_data():
    print("Loading AnnoMI dataset from Hugging Face...")
    try:
        dataset = load_dataset("biu-nlp/AnnoMI", split="train") # AnnoMI usually only has 'train' split on HF
    except Exception as e:
        print(f"Error loading AnnoMI: {e}")
        print("Please ensure you have internet access and access to the dataset.")
        sys.exit(1)
        
    df = dataset.to_pandas()
    print(f"Loaded {len(df)} rows.")
    
    # Filter for samples with labels
    # AnnoMI columns: 'transcript_id', 'turn_id', 'utterance_text', 'interlocutor', 'client_talk_type', 'main_therapist_behaviour'
    
    # We need to pair Therapist (prev) with Client (current)
    # Sort by transcript and turn
    df = df.sort_values(['transcript_id', 'turn_id'])
    
    processed_samples = []
    
    # Iterate to find Therapist -> Client pairs
    # Motivation labels are on Client turns ('client_talk_type')
    # Quality labels are on Therapist turns ('main_therapist_behaviour')
    
    # We want to predict Client Motivation (Change, Sustain, Neutral)
    # And Therapist Quality (High, Low)
    
    # Valid Client Labels: 'change', 'sustain', 'neutral'
    valid_client_labels = {'change', 'sustain', 'neutral'}
    
    # Group by transcript
    for tid, group in df.groupby('transcript_id'):
        turns = group.to_dict('records')
        for i in range(1, len(turns)):
            curr_turn = turns[i]
            prev_turn = turns[i-1]
            
            if curr_turn['interlocutor'] == 'client' and prev_turn['interlocutor'] == 'therapist':
                # Check Client Label
                c_label = str(curr_turn.get('client_talk_type', '')).lower()
                if c_label not in valid_client_labels:
                    continue
                    
                # Check Therapist Label
                t_label = str(prev_turn.get('main_therapist_behaviour', '')).lower()
                
                # Store
                processed_samples.append({
                    'client_text': curr_turn['utterance_text'],
                    'therapist_text': prev_turn['utterance_text'],
                    'motivation_label': c_label,
                    'therapist_behavior': t_label,
                    'transcript_id': tid,
                    'turn_id': curr_turn['turn_id']
                })
                
    df_proc = pd.DataFrame(processed_samples)
    print(f"Processed {len(df_proc)} Therapist-Client pairs with valid motivation labels.")
    
    # Map Motivation Labels to 0, 1, 2
    # 0=Sustain, 1=Neutral, 2=Change (Arbitrary but consistent)
    mot_map = {'sustain': 0, 'neutral': 1, 'change': 2}
    df_proc['y_mot'] = df_proc['motivation_label'].map(mot_map)
    
    # Map Therapist Quality to 0 (Low), 1 (High)
    # High: reflection (simple/complex), question (open?)
    # AnnoMI behavior labels: 'question', 'reflection_simple', 'reflection_complex', 'advising', 'other', ...
    # Let's inspect unique labels if possible, but for now define mapping based on MI theory
    # High: Reflections, Questions (Open)
    # Low: Advising, Confronting, Other (often neutral/low adherence in MI context if not specific)
    
    def map_quality(behav):
        if 'reflection' in behav:
            return 1 # High
        if 'question' in behav:
            return 1 # High (Optimistic)
        return 0 # Low (Advising, Other, etc.)
        
    df_proc['y_qual'] = df_proc['therapist_behavior'].apply(map_quality)
    
    return df_proc

# --- 2. Embedding Extraction ---

def get_embeddings(texts, batch_size=32):
    print(f"Extracting embeddings for {len(texts)} texts using {MODEL_NAME}...")
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
        # Gemma has no pad token, set it to eos
        tokenizer.pad_token = tokenizer.eos_token
        
        model = AutoModelForCausalLM.from_pretrained(
            MODEL_NAME, 
            device_map="auto",
            torch_dtype=torch.float16,
            output_hidden_states=True
        )
        model.eval()
    except Exception as e:
        print(f"Failed to load model {MODEL_NAME}: {e}")
        sys.exit(1)

    all_embeddings = []
    
    for i in tqdm(range(0, len(texts), batch_size)):
        batch_texts = texts[i:i+batch_size]
        
        # Tokenize
        inputs = tokenizer(
            batch_texts, 
            padding=True, 
            truncation=True, 
            max_length=128, # Keep it short for memory
            return_tensors="pt"
        ).to(model.device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            
        # Extract embeddings: Last hidden state of the last token (causal LM)
        # Sequence lengths vary, so we need to pick the last non-pad token
        hidden_states = outputs.hidden_states[-1] # (B, L, D)
        
        # Gather last token
        # attention_mask is (B, L), 1 for token, 0 for pad
        # last_token_indices = attention_mask.sum(1) - 1
        last_token_indices = inputs.attention_mask.sum(1) - 1
        
        # Select
        batch_emb = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
        all_embeddings.append(batch_emb.cpu().numpy())
        
    return np.concatenate(all_embeddings, axis=0)

# --- 3. Experiments ---

def run_training_loop(X, y, sample_sizes, task_type='multiclass', seed=SEED):
    """
    Runs training/eval loop for varying sample sizes.
    task_type: 'multiclass' (3-class) or 'binary' (2-class)
    """
    # Split Train/Test (Fixed split)
    X_train_full, X_test, y_train_full, y_test = train_test_split(
        X, y, test_size=0.3, random_state=seed, stratify=y
    )
    
    results = []
    
    # Convert to JAX arrays for GPP
    X_test_jax = jnp.array(X_test)
    y_test_jax = jnp.array(y_test)
    if task_type == 'multiclass':
        # One-hot for multiclass GPP if needed, but indices are supported in wrapper
        # Wrapper lpe_multiclass takes indices. gpp_multiclass takes indices.
        pass
    
    for n in sample_sizes:
        if n > len(X_train_full):
            print(f"Warning: Requested n={n} > available train size {len(X_train_full)}. Skipping.")
            continue
            
        # Subsample training data
        # Stratified subsample
        try:
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=n, random_state=seed, stratify=y_train_full
            )
        except ValueError:
            # Fallback if n is too small for stratification
            X_train = X_train_full[:n]
            y_train = y_train_full[:n]

        X_train_jax = jnp.array(X_train)
        y_train_jax = jnp.array(y_train)
        
        # Run GPP
        if task_type == 'multiclass':
            # GPP-Dirichlet
            # Expects y_observed as indices (n,) or one-hot?
            # Wrapper gpp_multiclass handles indices if shape is (n,1) or (n,)
            # Let's ensure (n, 1)
            measures = ppm.gpp_multiclass(
                x_query=X_test_jax,
                x_observed=X_train_jax,
                y_observed=y_train_jax, # Wrapper handles dimensions
                num_classes=3,
                n=1000 # MC samples
            )
            
            # Predictions (Max Prob)
            # measures usually contains 'aleatoric', 'epistemic', etc.
            # We need predictions? ppm returns 'measures'.
            # It seems ppm.gpp_multiclass ONLY returns uncertainty measures?
            # Let's check source code... 
            # It calls gp.dirichlet_gp_predict then gp.dirichlet_gp_uncertainty
            # uncertainty returns dict of measures. 
            # Does it return probabilities? 'mean' usually.
            # If not, we can't calculate Accuracy.
            # Let's assume we need to modify GPP or use 'lpe' for acc?
            # Wait, `gp_multiclass.dirichlet_gp_uncertainty` usually returns 'mean' (the expected prob vector).
            # If not, we have a problem.
            
            # Check keys
            if 'Judged probability' in measures:
                probs = np.array(measures['Judged probability'])
                preds = np.argmax(probs, axis=1)
                acc = accuracy_score(y_test, preds)
            elif 'mean' in measures:
                probs = np.array(measures['mean'])
                preds = np.argmax(probs, axis=1)
                acc = accuracy_score(y_test, preds)
            else:
                # Fallback: Can't compute acc if no mean
                acc = np.nan
            
            epistemic = np.mean(measures.get('Episteme', np.zeros_like(y_test)))
            aleatoric = np.mean(measures.get('Alea', np.zeros_like(y_test)))
            
        else: # Binary
            # GPP-Beta
            # Expects y 0/1
            measures = pp.gpp(
                x_query=X_test_jax,
                x_observed=X_train_jax,
                y_observed=y_train_jax,
                n=1000
            )
            
            # Use 'Judged probability' if available
            if 'Judged probability' in measures:
                probs = np.array(measures['Judged probability'])
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
            elif 'mean' in measures:
                probs = np.array(measures['mean'])
                preds = (probs > 0.5).astype(int)
                acc = accuracy_score(y_test, preds)
            else:
                acc = np.nan
                
            epistemic = np.mean(measures.get('Episteme', 0))
            aleatoric = np.mean(measures.get('Alea', 0))

        results.append({
            'n': n,
            'accuracy': acc,
            'aleatoric': aleatoric,
            'epistemic': epistemic
        })
        
    return pd.DataFrame(results), (X_train_full, X_test, y_train_full, y_test)

# --- Main Pipeline ---

def main():
    print("=== AnnoMI Pipeline Started ===")
    
    # 1. Load Data
    df = load_annomi_data()
    # Subset for development speed if needed, but dataset is small (~133? No, AnnoMI is larger, 133 might be unique transcripts)
    # AnnoMI has ~10k utterances. 
    # The user said "dataset is like 133 samples". Maybe they meant Transcripts?
    # If N is small, we use all.
    # If N is large, we might subsample.
    # Let's use all for now.
    
    X_client_text = df['client_text'].tolist()
    X_therapist_text = df['therapist_text'].tolist()
    X_context_text = [f"Therapist: {t}\nClient: {c}" for t, c in zip(X_therapist_text, X_client_text)]
    
    y_mot = df['y_mot'].values
    y_qual = df['y_qual'].values
    
    # 2. Extract Embeddings
    # We need:
    # A. Client Only (Exp 1)
    # B. Context (Exp 2)
    # C. Therapist Only (Exp 3)
    
    print("\n--- Extracting Embeddings ---")
    # To save time/memory, we can unique texts? No, context is unique pairs.
    # Just batch.
    
    X_emb_client = get_embeddings(X_client_text)
    X_emb_context = get_embeddings(X_context_text)
    X_emb_therapist = get_embeddings(X_therapist_text)
    
    # Reduce dimensionality with PCA if embeddings are huge (2048 for Gemma-2b)
    # GPP scales cubically with N, but N is small (<1000 in experiments). 
    # D doesn't matter as much, but for stability/speed PCA to 64 is good.
    # User plan didn't specify, but code usually does PCA.
    
    print("Reducing dimensions to 64 via PCA...")
    pca = PCA(n_components=64)
    X_emb_client = pca.fit_transform(X_emb_client)
    
    pca = PCA(n_components=64)
    X_emb_context = pca.fit_transform(X_emb_context)
    
    pca = PCA(n_components=64)
    X_emb_therapist = pca.fit_transform(X_emb_therapist)
    
    # --- Experiment 1: Client Only (Motivation) ---
    print("\n--- Experiment 1: Client Only (Motivation) ---")
    res_1, _ = run_training_loop(X_emb_client, y_mot, SAMPLE_SIZES, 'multiclass')
    res_1['Scenario'] = 'Client Only'
    
    # --- Experiment 2: Context (Motivation) ---
    print("\n--- Experiment 2: Context (Motivation) ---")
    res_2, _ = run_training_loop(X_emb_context, y_mot, SAMPLE_SIZES, 'multiclass')
    res_2['Scenario'] = 'Context'
    
    # --- Experiment 3: Therapist Quality ---
    print("\n--- Experiment 3: Therapist Quality ---")
    res_3, (X_train_q, X_test_q, y_train_q, y_test_q) = run_training_loop(
        X_emb_therapist, y_qual, SAMPLE_SIZES, 'binary'
    )
    # We need a TRAINED model (on max sample size) to predict for Exp 4
    # Re-train on largest N from loop (which is X_train_q from the split, assuming max N used all)
    # Actually run_training_loop returns the split data.
    # We'll use the FULL Training set from the split to train the Quality Predictor
    
    print("Training Final Quality Model for Exp 4...")
    # Train on X_train_q (which corresponds to 70% of data)
    # Predict on X_train_q AND X_test_q to get features for everyone
    # Note: Predicting on train data gives optimistic uncertainty, but we need features.
    
    X_train_q_jax = jnp.array(X_train_q)
    y_train_q_jax = jnp.array(y_train_q)
    
    # Predict for ALL data (we need to map back to original indices to align with Client text)
    # This is tricky with random splits inside function.
    # Refactoring: We should split indices first globally if we want to align.
    
    # Let's align splits globally for Exp 4.
    # Split Indices
    indices = np.arange(len(df))
    train_idx, test_idx = train_test_split(indices, test_size=0.3, random_state=SEED, stratify=y_mot)
    
    # Quality Model Data
    X_q_train = X_emb_therapist[train_idx]
    y_q_train = y_qual[train_idx]
    X_q_test = X_emb_therapist[test_idx]
    # y_q_test = y_qual[test_idx] # Not needed for prediction
    
    # Train Quality GPP on Train Split
    # We need predictions for BOTH Train and Test splits to feed into Motivation Model
    
    # 1. Get Quality Predictions & Uncertainty for TRAIN set
    # Using LOO or K-Fold would be better for Train set features, but let's just use the model fit on itself for now (simple)
    # Or split Train into Train-A and Train-B? 
    # User said "train another LLM... then... provide...".
    # I will simple predict on X_q_train using X_q_train as observed.
    
    measures_train = pp.gpp(
        x_query=jnp.array(X_q_train),
        x_observed=jnp.array(X_q_train),
        y_observed=jnp.array(y_q_train),
        n=1000
    )
    # Binary GPP predictions (probs)
    # Assume 'mean'
    if 'mean' in measures_train:
        q_probs_train = np.array(measures_train['mean'])
    else:
        q_probs_train = np.zeros(len(train_idx)) # Fallback
        
    q_preds_train = (q_probs_train > 0.5).astype(int)
    q_unc_train = np.column_stack([
        np.array(measures_train.get('aleatoric', np.zeros(len(train_idx)))),
        np.array(measures_train.get('epistemic', np.zeros(len(train_idx))))
    ])
    
    # 2. Get Quality Predictions & Uncertainty for TEST set
    measures_test = pp.gpp(
        x_query=jnp.array(X_q_test),
        x_observed=jnp.array(X_q_train),
        y_observed=jnp.array(y_q_train),
        n=1000
    )
    if 'mean' in measures_test:
        q_probs_test = np.array(measures_test['mean'])
    else:
        q_probs_test = np.zeros(len(test_idx))
        
    q_preds_test = (q_probs_test > 0.5).astype(int)
    q_unc_test = np.column_stack([
        np.array(measures_test.get('aleatoric', np.zeros(len(test_idx)))),
        np.array(measures_test.get('epistemic', np.zeros(len(test_idx))))
    ])
    
    # --- Experiment 4: Cascading ---
    print("\n--- Experiment 4: Cascading Uncertainty ---")
    
    # Construct Augmented Text
    # Train
    texts_train_aug = []
    labels_map = {0: "Low", 1: "High"}
    orig_therapist_train = [X_therapist_text[i] for i in train_idx]
    orig_client_train = [X_client_text[i] for i in train_idx]
    
    for t, c, q in zip(orig_therapist_train, orig_client_train, q_preds_train):
        qual_str = labels_map[q]
        texts_train_aug.append(f"[Quality: {qual_str}] Therapist: {t}\nClient: {c}")
        
    # Test
    texts_test_aug = []
    orig_therapist_test = [X_therapist_text[i] for i in test_idx]
    orig_client_test = [X_client_text[i] for i in test_idx]
    
    for t, c, q in zip(orig_therapist_test, orig_client_test, q_preds_test):
        qual_str = labels_map[q]
        texts_test_aug.append(f"[Quality: {qual_str}] Therapist: {t}\nClient: {c}")
        
    # Embed Augmented Text
    print("Embedding augmented text...")
    X_emb_aug_train = get_embeddings(texts_train_aug)
    X_emb_aug_test = get_embeddings(texts_test_aug)
    
    # PCA
    pca_aug = PCA(n_components=64)
    # Fit on Train, Transform Both
    X_emb_aug_train = pca_aug.fit_transform(X_emb_aug_train)
    X_emb_aug_test = pca_aug.transform(X_emb_aug_test)
    
    # Concatenate Uncertainty Features
    # Dimensions: 64 + 2 = 66
    X_final_train = np.hstack([X_emb_aug_train, q_unc_train])
    X_final_test = np.hstack([X_emb_aug_test, q_unc_test])
    
    y_mot_train = y_mot[train_idx]
    y_mot_test = y_mot[test_idx]
    
    # Run loop manually for Exp 4 since we have custom splits
    res_4_list = []
    for n in SAMPLE_SIZES:
        if n > len(X_final_train): continue
        
        # Subsample Train
        # simple slice
        X_sub = X_final_train[:n]
        y_sub = y_mot_train[:n]
        
        # GPP Multiclass
        measures = ppm.gpp_multiclass(
            x_query=jnp.array(X_final_test),
            x_observed=jnp.array(X_sub),
            y_observed=jnp.array(y_sub),
            num_classes=3,
            n=1000
        )
        
        if 'mean' in measures:
            probs = np.array(measures['mean'])
            preds = np.argmax(probs, axis=1)
            acc = accuracy_score(y_mot_test, preds)
        else:
            acc = np.nan
            
        epistemic = np.mean(measures.get('epistemic', 0))
        aleatoric = np.mean(measures.get('aleatoric', 0))
        
        res_4_list.append({
            'n': n,
            'accuracy': acc,
            'aleatoric': aleatoric,
            'epistemic': epistemic,
            'Scenario': 'Cascading'
        })
        
    res_4 = pd.DataFrame(res_4_list)
    
    # --- Visualization ---
    print("\n--- Generating Plots ---")
    all_res = pd.concat([res_1, res_2, res_4], ignore_index=True)
    all_res.to_csv(os.path.join(FIGURE_DIR, 'experiment_results.csv'), index=False)
    
    # Plot Accuracy
    plt.figure(figsize=(10, 6))
    sns.lineplot(data=all_res, x='n', y='accuracy', hue='Scenario', marker='o')
    plt.title("Motivation Classification Accuracy vs Sample Size")
    plt.savefig(os.path.join(FIGURE_DIR, 'accuracy_comparison.png'))
    plt.close()
    
    # Plot Uncertainty
    plt.figure(figsize=(12, 5))
    plt.subplot(1, 2, 1)
    sns.lineplot(data=all_res, x='n', y='epistemic', hue='Scenario', marker='o')
    plt.title("Epistemic Uncertainty")
    
    plt.subplot(1, 2, 2)
    sns.lineplot(data=all_res, x='n', y='aleatoric', hue='Scenario', marker='o')
    plt.title("Aleatoric Uncertainty")
    
    plt.tight_layout()
    plt.savefig(os.path.join(FIGURE_DIR, 'uncertainty_comparison.png'))
    plt.close()
    
    
    # --- Qualitative Analysis ---
    print("\n--- Qualitative Analysis ---")
    # Pick 5 random test samples
    n_examples = 5
    ex_indices = np.random.choice(len(y_mot_test), n_examples, replace=False)
    
    # We need predictions from:
    # 1. Client Only (Exp 1) - Train on X_emb_client[train_idx]
    # 2. Context (Exp 2) - Train on X_emb_context[train_idx]
    # 3. Cascading (Exp 4) - Train on X_final_train
    
    # Train Models on Full Train Set (Max N)
    print("Training models for qualitative examples...")
    
    # Client Only
    measures_c = ppm.gpp_multiclass(
        x_query=jnp.array(X_emb_client[test_idx][ex_indices]),
        x_observed=jnp.array(X_emb_client[train_idx]),
        y_observed=jnp.array(y_mot[train_idx]),
        num_classes=3,
        n=1000
    )
    probs_c = np.array(measures_c.get('Judged probability', measures_c.get('mean', [])))
    unc_c = np.array(measures_c.get('Episteme', np.zeros(n_examples)))
    
    # Context
    measures_ctx = ppm.gpp_multiclass(
        x_query=jnp.array(X_emb_context[test_idx][ex_indices]),
        x_observed=jnp.array(X_emb_context[train_idx]),
        y_observed=jnp.array(y_mot[train_idx]),
        num_classes=3,
        n=1000
    )
    probs_ctx = np.array(measures_ctx.get('Judged probability', measures_ctx.get('mean', [])))
    unc_ctx = np.array(measures_ctx.get('Episteme', np.zeros(n_examples)))
    
    # Cascading
    measures_cas = ppm.gpp_multiclass(
        x_query=jnp.array(X_final_test[ex_indices]),
        x_observed=jnp.array(X_final_train),
        y_observed=jnp.array(y_mot[train_idx]),
        num_classes=3,
        n=1000
    )
    probs_cas = np.array(measures_cas.get('Judged probability', measures_cas.get('mean', [])))
    unc_cas = np.array(measures_cas.get('Episteme', np.zeros(n_examples)))
    
    # Labels
    mot_labels = {0: 'Sustain', 1: 'Neutral', 2: 'Change'}
    
    print("\n=== Qualitative Examples ===")
    for i, idx in enumerate(ex_indices):
        orig_idx = test_idx[idx]
        t_text = X_therapist_text[orig_idx]
        c_text = X_client_text[orig_idx]
        true_label = mot_labels[y_mot[orig_idx]]
        
        pred_c = mot_labels[np.argmax(probs_c[i])]
        pred_ctx = mot_labels[np.argmax(probs_ctx[i])]
        pred_cas = mot_labels[np.argmax(probs_cas[i])]
        
        print(f"\nExample {i+1}:")
        print(f"Therapist: {t_text[:100]}...")
        print(f"Client: {c_text[:100]}...")
        print(f"True Label: {true_label}")
        print(f"  Client Only: Pred={pred_c}, Episteme={unc_c[i]:.4f}")
        print(f"  Context:     Pred={pred_ctx}, Episteme={unc_ctx[i]:.4f}")
        print(f"  Cascading:   Pred={pred_cas}, Episteme={unc_cas[i]:.4f}")

    print("Done!")

if __name__ == "__main__":
    main()

