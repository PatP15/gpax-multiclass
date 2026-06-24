import os
import sys
import gc
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score
from datasets import load_dataset
import torch

from transformers import AutoTokenizer, AutoModelForCausalLM, AutoModelForVision2Seq
from tqdm import tqdm

# Add repo root (the dir containing GPax/) to sys.path, robust to this file's location.
def _find_repo_root(start):
    d = os.path.dirname(os.path.abspath(start))
    while d != os.path.dirname(d):
        if os.path.isdir(os.path.join(d, 'GPax')):
            return d
        d = os.path.dirname(d)
    return os.path.dirname(os.path.abspath(start))

_REPO_ROOT = _find_repo_root(__file__)
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
_HERE = os.path.dirname(os.path.abspath(__file__))

# --- Constants & Configuration ---
SEED = 42
SAMPLE_SIZES = [50, 100, 250, 500, 1000, 1500, 2000, 2400]

def get_model_config():
    model_type = os.environ.get("ANNOMI_MODEL_TYPE", "qwen").lower()
    
    if model_type == "qwen":
        model_name = "Qwen/Qwen3-VL-30B-A3B-Thinking"
        subfolder = "qwen"
    elif model_type == "gemma":
        model_name = "google/gemma-3-27b-it"
        subfolder = "gemma"
    else: 
        model_name = "Qwen/Qwen3-VL-30B-A3B-Thinking"
        subfolder = "qwen"
        
    return model_name, subfolder, model_type

MODEL_NAME, MODEL_SUBFOLDER, MODEL_TYPE = get_model_config()

# New Binary Directories
BASE_DIR = _HERE
DATA_DIR = os.path.join(BASE_DIR, "data", MODEL_SUBFOLDER)
RESULTS_DIR = os.path.join(BASE_DIR, "results", MODEL_SUBFOLDER)
FIGURE_DIR = os.path.join(BASE_DIR, "figures", MODEL_SUBFOLDER)

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(FIGURE_DIR, exist_ok=True)

def process_single_df(df, name="dataset"):
    # Filter and Process
    valid_client_labels = {'change', 'sustain', 'neutral'}
    processed_samples = []
    
    # Sort by transcript and timestamp (or utterance_id which is safer)
    if 'transcript_id' in df.columns and 'utterance_id' in df.columns:
        df = df.sort_values(['transcript_id', 'utterance_id'])
    else:
        print(f"Warning: {name} missing sort columns, using index order.")
    
    # Group by transcript to find pairs
    for tid, group in df.groupby('transcript_id'):
        turns = group.to_dict('records')
        for i in range(1, len(turns)):
            curr_turn = turns[i]
            prev_turn = turns[i-1]
            
            if curr_turn['interlocutor'] == 'client' and prev_turn['interlocutor'] == 'therapist':
                c_label = str(curr_turn.get('client_talk_type', '')).lower()
                if c_label not in valid_client_labels:
                    continue
                    
                t_label = str(prev_turn.get('main_therapist_behaviour', '')).lower()
                if t_label == 'nan': t_label = 'other'
                
                processed_samples.append({
                    'client_text': curr_turn['utterance_text'],
                    'therapist_text': prev_turn['utterance_text'],
                    'motivation_label': c_label,
                    'therapist_behavior': t_label,
                    'transcript_id': tid,
                    'turn_id': curr_turn['utterance_id']
                })
                
    df_proc = pd.DataFrame(processed_samples)
    print(f"Processed {name}: {len(df_proc)} pairs.")
    
    if df_proc.empty:
        return df_proc
        
    # Map Labels - BINARY MODIFICATION
    # Sustain (0) + Neutral (1) -> Non-Change (0)
    # Change (2) -> Change (1)
    mot_map = {'sustain': 0, 'neutral': 0, 'change': 1}
    df_proc['y_mot'] = df_proc['motivation_label'].map(mot_map)
    
    def map_quality(behav):
        if 'reflection' in behav: return 1
        if 'question' in behav: return 1 
        return 0
        
    df_proc['y_qual'] = df_proc['therapist_behavior'].apply(map_quality)
    
    return df_proc

def load_annomi_data():
    print("Loading AnnoMI datasets from local CSVs...")
    base_data_dir = os.path.join(os.path.dirname(_HERE), "data")
    train_path = os.path.join(base_data_dir, "IC_AnnoMI.csv")
    test_path = os.path.join(base_data_dir, "IC-AnnoMI (test set).csv")
    
    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print(f"Error: Files not found.\nTrain: {train_path}\nTest: {test_path}")
        sys.exit(1)
        
    try:
        # Load with semicolon separator
        df_train = pd.read_csv(train_path, sep=';')
        df_test = pd.read_csv(test_path, sep=';')
    except Exception as e:
        print(f"Error loading CSVs: {e}")
        sys.exit(1)
        
    print(f"Loaded Train: {len(df_train)} rows")
    print(f"Loaded Test: {len(df_test)} rows")
    
    df_train_proc = process_single_df(df_train, "Train")
    df_test_proc = process_single_df(df_test, "Test")
    
    return df_train_proc, df_test_proc

def load_model(model_name=MODEL_NAME):
    print(f"Loading model {model_name} (Type: {MODEL_TYPE})...")
    
    try:
        with open(os.path.join(_REPO_ROOT, "huggingfacetoken.txt"), "r") as f:
            hf_token = f.read().strip()
    except FileNotFoundError:
        hf_token = True

    try:
        if MODEL_TYPE == "gemma":
            # User requested pipeline for Gemma 3
            from transformers import pipeline
            pipe = pipeline(
                "image-text-to-text",
                model=model_name,
                device="cuda" if torch.cuda.is_available() else "cpu",
                torch_dtype=torch.bfloat16,
                token=hf_token
            )
            model = pipe.model
            tokenizer = pipe.tokenizer
            model.eval()
            return model, tokenizer
            
        elif MODEL_TYPE == "qwen":
            try:
                from transformers import Qwen2VLForConditionalGeneration # Fallback or similar
                try:
                    from transformers import Qwen3VLMoeForConditionalGeneration
                    model_class = Qwen3VLMoeForConditionalGeneration
                except ImportError:
                    print("Warning: Qwen3VLMoeForConditionalGeneration not found, trying AutoModel.")
                    model_class = AutoModelForVision2Seq
                    
                tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
                model = model_class.from_pretrained(
                    model_name,
                    dtype="auto",
                    attn_implementation="sdpa",
                    token=hf_token,
                    trust_remote_code=True,
                    device_map="auto"
                )
                # model.to("cuda" if torch.cuda.is_available() else "cpu")
            except Exception as e:
                    print(f"Error loading Qwen specific class: {e}. Fallback to AutoModel.")
                    tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token, trust_remote_code=True)
                    model = AutoModelForCausalLM.from_pretrained(
                        model_name, 
                        token=hf_token, 
                        trust_remote_code=True, 
                        dtype="auto",
                        device_map="auto"
                    )
                    # model.to("cuda" if torch.cuda.is_available() else "cpu")
            return model, tokenizer
            
        else: # Generic Fallback
            tokenizer = AutoTokenizer.from_pretrained(model_name, token=hf_token)
            tokenizer.pad_token = tokenizer.eos_token
            
            model = AutoModelForCausalLM.from_pretrained(
                model_name, 
                device_map="auto",
                torch_dtype=torch.float16, 
                output_hidden_states=True,
                token=hf_token,
                low_cpu_mem_usage=True,
                offload_folder="offload"
            )
            # model.to("cuda" if torch.cuda.is_available() else "cpu") # handled by device_map="auto"
            return model, tokenizer

    except Exception as e:
        print(f"Failed to load model {model_name}: {e}")
        sys.exit(1)

def get_embeddings(texts, model=None, tokenizer=None, batch_size=1, model_name=MODEL_NAME, max_length=256):
    print(f"Extracting embeddings for {len(texts)} texts using {model_name} (max_len={max_length})...")
    
    should_cleanup = False
    if model is None or tokenizer is None:
        should_cleanup = True
        model, tokenizer = load_model(model_name)

    # Ensure model has output_hidden_states enabled
    if hasattr(model.config, "output_hidden_states"):
        model.config.output_hidden_states = True

    all_embeddings = []
    
    try:
        for i in tqdm(range(0, len(texts), batch_size)):
            batch_texts = texts[i:i+batch_size]
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
                
            inputs = tokenizer(batch_texts, padding=True, truncation=True, max_length=max_length, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                # Some models might need special kwargs, but most HF models allow **inputs with input_ids/attention_mask
                if MODEL_TYPE == "qwen" or (MODEL_TYPE == "gemma" and "image" in model_name):
                     outputs = model(**inputs, output_hidden_states=True)
                else:
                    outputs = model(**inputs, output_hidden_states=True)
            
            hidden_states = outputs.hidden_states[-1]
            last_token_indices = inputs.attention_mask.sum(1) - 1
            batch_emb = hidden_states[torch.arange(hidden_states.size(0)), last_token_indices]
            
            # Convert to float32 before numpy
            all_embeddings.append(batch_emb.float().cpu().numpy())
            
            del inputs, outputs, hidden_states, batch_emb
            torch.cuda.empty_cache()
            
    finally:
        if should_cleanup:
            del model
            del tokenizer
            torch.cuda.empty_cache()
            gc.collect()
        
    return np.concatenate(all_embeddings, axis=0)

def balance_dataset(X, y, seed=SEED):
    """
    Undersample majority classes to match the size of the minority class (or specified logic).
    For binary (Change vs Non-Change), we want Non-Change >= Change.
    """
    unique_classes, counts = np.unique(y, return_counts=True)
    # Binary: 0=Non-Change, 1=Change
    
    # Check counts
    count_0 = counts[0] # Non-Change
    count_1 = counts[1] # Change
    
    print(f"Original counts: Non-Change={count_0}, Change={count_1}")
    
    # Logic: Balance Trainset based on max number of Change speak
    # "the combination of neutral and sustain should be the same or larger than the change class"
    # This implies we can undersample Non-Change to be equal to Change, OR keep it larger.
    # Standard balancing usually implies making them equal. 
    # Let's target equal size = count_1 (Change count).
    
    target_size = count_1
    print(f"Balancing dataset: Reducing Non-Change to match Change ({target_size} samples).")
    
    indices = []
    rng = np.random.default_rng(seed)
    
    # Process Class 0 (Non-Change) - Undersample
    idx_0 = np.where(y == 0)[0]
    if len(idx_0) > target_size:
        sel_0 = rng.choice(idx_0, size=target_size, replace=False)
    else:
        sel_0 = idx_0 # Should not happen given logic, but safe fallback
    indices.append(sel_0)
    
    # Process Class 1 (Change) - Keep All
    idx_1 = np.where(y == 1)[0]
    indices.append(idx_1) # Keep all Change samples
        
    all_indices = np.concatenate(indices)
    rng.shuffle(all_indices) 
    
    if isinstance(X, np.ndarray):
        return X[all_indices], y[all_indices]
    else:
        return [X[i] for i in all_indices], y[all_indices]

def run_training_loop(X_train_full, y_train_full, X_test, y_test, sample_sizes, task_type='binary', seed=SEED, return_raw=False, method='gpp', balance=True):
    # Lazy import JAX and GPax here to avoid initialization in Step 1
    import jax
    import jax.numpy as jnp
    from GPax.probing import probabilistic_probe_multiclass as ppm
    from GPax.probing import probabilistic_probe as pp
    
    # 1. Balance Training Dataset (Undersampling)
    if balance:
        print("Balancing Training Set (Binary)...")
        X_train_full, y_train_full = balance_dataset(X_train_full, y_train_full, seed=seed)

    results = []
    raw_results = {} 
    
    X_test_jax = jnp.array(X_test)
    y_test_jax = jnp.array(y_test)
    
    for n in sample_sizes:
        if n > len(X_train_full):
            continue
            
        try:
            # Subsample n from the balanced training set
            X_train, _, y_train, _ = train_test_split(
                X_train_full, y_train_full, train_size=n, random_state=seed, stratify=y_train_full
            )
        except ValueError:
            X_train = X_train_full[:n]
            y_train = y_train_full[:n]

        X_train_jax = jnp.array(X_train)
        y_train_jax = jnp.array(y_train)
        
        # --- Binary (Change vs Non-Change) ---
        # Note: Even though it's binary, we can use multiclass code with num_classes=2 or the binary implementation.
        # The prompt implies we want to compare with previous logic.
        # Let's use binary implementation pp.gpp / pp.lpe for consistency with typical binary tasks.
        
        if method == 'gpp':
            measures = pp.gpp(
                x_query=X_test_jax, x_observed=X_train_jax, y_observed=y_train_jax, n=1000
            )
        elif method == 'lpe':
            measures = pp.lpe(
                x_query=X_test_jax, x_observed=X_train_jax, y_observed=y_train_jax, repeats=100
            )
        else:
            raise ValueError(f"Unknown method: {method}")

        if 'Judged probability' in measures:
            probs_pos = np.array(measures['Judged probability'])
        elif 'mean' in measures:
            probs_pos = np.array(measures['mean'])
        else:
            probs_pos = np.zeros(len(y_test))
            
        preds = (probs_pos > 0.5).astype(int)
        probs = np.vstack([1-probs_pos, probs_pos]).T 
        
        epistemic = np.array(measures.get('Episteme', np.zeros(len(y_test))))
        aleatoric = np.array(measures.get('Alea', np.zeros(len(y_test))))

        # Metrics
        acc = accuracy_score(y_test, preds)
        
        # AUROC (Misclassification Detection)
        misclassified = (preds != y_test).astype(int)
        
        if len(np.unique(misclassified)) == 2:
            # 'Episteme' is a CONFIDENCE (higher = more certain); misclassification
            # correlates with UNCERTAINTY, so the score must be negated.
            auroc_mis = roc_auc_score(misclassified, -epistemic)
        else:
            auroc_mis = np.nan

        results.append({
            'n': n,
            'accuracy': acc,
            'aleatoric': np.mean(aleatoric),
            'epistemic': np.mean(epistemic),
            'auroc': auroc_mis
        })
        
        raw_results[n] = {
            'y_true': y_test,
            'y_pred': preds,
            'probs': probs,
            'epistemic': epistemic,
            'aleatoric': aleatoric
        }
        
    df_res = pd.DataFrame(results)
    
    if return_raw:
        return df_res, raw_results
    else:
        return df_res, (X_train_full, X_test, y_train_full, y_test)


