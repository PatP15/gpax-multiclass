import h5py
import numpy as np
import jax
import jax.numpy as jnp
from GPax.probing import gp
from GPax.probing import gp_multiclass
import sys
import os

# Set JAX to CPU to avoid potential memory/bus errors in restricted environments
os.environ['JAX_PLATFORM_NAME'] = 'cpu'

def verify_equivalence():
    print("Starting verification...", flush=True)
    
    # Try to load 3dshapes data
    X_train, y_train, X_query = None, None, None
    
    try:
        if os.path.exists('3dshapes.h5'):
            print("Found 3dshapes.h5, attempting to load subset...", flush=True)
            f = h5py.File('3dshapes.h5', 'r')
            labels = f['labels']
            # Read only a small chunk of labels to find indices
            # There are 480k images. Reading all labels (480k, 6) floats is fine (~23MB)
            labels_all = labels[:] 
            
            # Factor 4 is shape. 0=Cube, 1=Cylinder.
            mask = np.isin(labels_all[:, 4], [0, 1])
            indices = np.where(mask)[0]
            
            # Select 20 random samples for train, 10 for query
            np.random.seed(42)
            selected_indices = np.random.choice(indices, 30, replace=False)
            selected_indices.sort()
            
            # Load images one by one or in small batches
            images_dset = f['images']
            X_list = []
            for idx in selected_indices:
                X_list.append(images_dset[idx])
            
            X = np.array(X_list).astype(np.float32) / 255.0
            X_flat = X.reshape(X.shape[0], -1)
            y_factors = labels_all[selected_indices]
            y_cls = y_factors[:, 4].astype(int)
            
            # Shuffle
            perm = np.random.permutation(len(X_flat))
            X_flat = X_flat[perm]
            y_cls = y_cls[perm]
            
            n_train = 20
            X_train = X_flat[:n_train]
            y_train = y_cls[:n_train]
            X_query = X_flat[n_train:]
            
            print(f"Successfully loaded real data. Train shape: {X_train.shape}", flush=True)
            f.close()
            
    except Exception as e:
        print(f"Could not load real data: {e}. Falling back to synthetic.", flush=True)
        
    if X_train is None:
        print("Generating synthetic data...", flush=True)
        n_samples = 30
        n_features = 50 
        np.random.seed(42)
        X_flat = np.random.randn(n_samples, n_features).astype(np.float32)
        y_cls = np.random.randint(0, 2, n_samples)
        
        n_train = 20
        X_train = X_flat[:n_train]
        y_train = y_cls[:n_train]
        X_query = X_flat[n_train:]

    print(f"Train Data Shape: {X_train.shape}")
    print(f"Classes: {np.unique(y_train)}")

    # Shared Parameters
    alpha_eps = 0.1
    strength = 10.0
    params = {
        'alpha_eps': alpha_eps,
        'strength': strength,
        'lengthscale': 10.0, # Adjusted for high dim
        'signal_variance': 1.0,
        'constant': 0.0 
    }
    
    # --- Run Binary Beta GP ---
    print("\n--- Running Binary Beta GP ---", flush=True)
    # y must be n x 1
    y_train_binary = y_train[:, None]
    
    mean_func = gp.constant_mean
    cov_func = gp.squared_exponential_kernel 
    
    params_binary = params.copy()
    
    try:
        predictions_binary = gp.beta_gp_predict(
            mean_func=mean_func,
            cov_func=cov_func,
            params=params_binary,
            x_query=X_query,
            x_observed=X_train,
            y_observed=y_train_binary,
            var_only=True
        )
        
        mu_binary, var_binary = gp.get_latent_gp(predictions_binary)
        print(f"Binary mu (first 5): {mu_binary.flatten()[:5]}", flush=True)
        
    except Exception as e:
        print(f"Binary GP Failed: {e}", flush=True)
        return
    
    # --- Run Multiclass Dirichlet GP ---
    print("\n--- Running Multiclass Dirichlet GP ---", flush=True)
    params_multi = params.copy()
    params_multi['num_classes'] = 2
    
    try:
        predictions_multi = gp_multiclass.dirichlet_gp_predict(
            mean_func=mean_func,
            cov_func=cov_func,
            params=params_multi,
            x_query=X_query,
            x_observed=X_train,
            y_observed=y_train_binary, 
            var_only=True
        )
        
        mus_multi, vars_multi = gp_multiclass.get_latent_gp_dirichlet(predictions_multi)
        
        # Reconstruct "Binary-like" mu and var from Multiclass outputs
        # Binary GP models f_positive - f_negative.
        # Multiclass GP has [f_0, f_1].
        # If Class 1 is Positive, then we expect match with f_1 - f_0.
        
        mu_multi_constructed = mus_multi[1] - mus_multi[0]
        var_multi_constructed = vars_multi[1] + vars_multi[0]
        
        print(f"Constructed Multi mu (first 5): {mu_multi_constructed.flatten()[:5]}", flush=True)
        
    except Exception as e:
        print(f"Multiclass GP Failed: {e}", flush=True)
        return
    
    # --- Comparison ---
    print("\n--- Comparison Results ---", flush=True)
    
    diff_mu = jnp.abs(mu_binary - mu_multi_constructed)
    max_diff_mu = jnp.max(diff_mu)
    print(f"Max difference in latent mean: {max_diff_mu}")
    
    diff_var = jnp.abs(var_binary - var_multi_constructed)
    max_diff_var = jnp.max(diff_var)
    print(f"Max difference in latent variance: {max_diff_var}")
    
    tol = 1e-4 # Relax slightly for float precision in complex operations
    if max_diff_mu < tol and max_diff_var < tol:
        print("\nSUCCESS: Multiclass GP matches Binary GP.")
    else:
        print("\nFAILURE: Discrepancy detected.")

if __name__ == "__main__":
    verify_equivalence()
