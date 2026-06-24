
import jax
import jax.numpy as jnp
import numpy as np
import pandas as pd
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm

def test_episteme_parity():
    print("=== Testing Episteme Parity (Binary vs Multiclass) ===")
    
    # 1. Generate Synthetic Binary Data
    key = jax.random.PRNGKey(42)
    N_obs = 50
    N_query = 10
    D = 5
    
    X_obs = jax.random.normal(key, (N_obs, D))
    # Simple linear boundary
    w = jax.random.normal(key, (D, 1))
    logits = X_obs @ w
    y_obs = (logits > 0).astype(int).flatten() # (N,)
    
    X_query = jax.random.normal(key, (N_query, D))

    # 2. Run Binary GPP
    print("\n--- Running Binary GPP ---")
    res_bin = pp.gpp(X_query, X_obs, y_obs, n=10000) # Reduced samples for speed
    ep_bin = res_bin['Episteme']
    jp_bin = res_bin['Judged probability']
    
    print(f"Binary Episteme (First 5): {ep_bin[:5]}")
    print(f"Binary Judged Prob (First 5): {jp_bin[:5]}")
    
    # 3. Run Multiclass GPP (2 classes)
    print("\n--- Running Multiclass GPP (2 classes) ---")
    y_obs_oh = jax.nn.one_hot(y_obs, 2)
    res_multi = ppm.gpp_multiclass(X_query, X_obs, y_obs_oh, num_classes=2, n=10000)
    
    # Multiclass outputs Episteme as (N,) array
    ep_multi = res_multi['Episteme']
    jp_multi = res_multi['categorical_mu'][:, 1] # Probability of class 1
    
    print(f"Multiclass Episteme (First 5): {ep_multi[:5]}")
    print(f"Multiclass Judged Prob Class 1 (First 5): {jp_multi[:5]}")
    
    # 4. Compare
    # Check Judged Probability Correlation
    corr_jp = np.corrcoef(jp_bin, jp_multi)[0, 1]
    print(f"\nJudged Prob Correlation: {corr_jp:.4f}")
    
    # Check Episteme Correlation
    ep_bin_flat = np.array(ep_bin).flatten()
    ep_multi_flat = np.array(ep_multi).flatten()
    corr_ep = np.corrcoef(ep_bin_flat, ep_multi_flat)[0, 1]
    print(f"Episteme Correlation (Original): {corr_ep:.4f}")
        
    # 5. Test Proposed Approximation
    # Re-run Multiclass to get raw variances if needed, or just calculate from samples in test?
    # Accessing internal samples is hard.
    # But we can check if InfoGain is anticorrelated.
    
    print(f"\nBinary Episteme Range: [{np.min(ep_bin):.4f}, {np.max(ep_bin):.4f}]")
    print(f"Multiclass Episteme Range: [{np.min(ep_multi):.4f}, {np.max(ep_multi):.4f}]")
    
    # Let's assume we implement the change.


if __name__ == "__main__":
    test_episteme_parity()
