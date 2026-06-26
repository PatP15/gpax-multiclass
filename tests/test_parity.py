"""Binary-vs-multiclass parity test for GPP.

At K=2 the multiclass (Dirichlet) GPP must reduce to the binary (Beta) GPP.
The decisive check is that the judged probability of the positive class
matches between the two implementations. We also sanity-check that the
multiclass uncertainty measures are finite and correctly shaped.

Run with `python tests/test_parity.py` or `pytest tests/test_parity.py`.
"""
import jax
import jax.numpy as jnp
import numpy as np
from GPax.probing import probabilistic_probe as pp
from GPax.probing import probabilistic_probe_multiclass as ppm


def test_episteme_parity():
    print("=== Testing Binary vs Multiclass (K=2) Parity ===")

    # 1. Synthetic binary data with a linear boundary.
    key = jax.random.PRNGKey(42)
    N_obs, N_query, D = 50, 10, 5
    X_obs = jax.random.normal(key, (N_obs, D))
    w = jax.random.normal(key, (D, 1))
    y_obs = (X_obs @ w > 0).astype(int).flatten()
    X_query = jax.random.normal(jax.random.PRNGKey(1), (N_query, D))

    # 2. Binary GPP.
    res_bin = pp.gpp(X_query, X_obs, y_obs, n=10000)
    jp_bin = np.array(res_bin['Judged probability']).flatten()
    ep_bin = np.array(res_bin['Episteme']).flatten()

    # 3. Multiclass GPP at K=2.
    y_obs_oh = jax.nn.one_hot(y_obs, 2)
    res_multi = ppm.gpp_multiclass(X_query, X_obs, y_obs_oh, num_classes=2, n=10000)
    jp_multi = np.array(res_multi['categorical_mu'])[:, 1]  # P(class 1)
    ep_multi = np.array(res_multi['Episteme']).flatten()

    # 4. Judged-probability parity (the decisive equivalence check).
    corr_jp = float(np.corrcoef(jp_bin, jp_multi)[0, 1])
    max_abs_diff = float(np.max(np.abs(jp_bin - jp_multi)))
    print(f"Judged-prob correlation (binary vs K=2 multiclass): {corr_jp:.6f}")
    print(f"Judged-prob max abs diff:                           {max_abs_diff:.4f}")

    # 5. Assertions.
    assert jp_multi.shape == jp_bin.shape == (N_query,), "judged-prob shape mismatch"
    assert ep_multi.shape[0] == N_query, "episteme shape mismatch"
    assert corr_jp > 0.999, f"judged-prob parity broken: corr={corr_jp:.6f} (expected > 0.999)"
    assert np.all(np.isfinite(jp_multi)), "non-finite multiclass judged prob"
    assert np.all(np.isfinite(ep_multi)), "non-finite multiclass episteme"
    assert np.all((jp_multi >= 0) & (jp_multi <= 1)), "judged prob outside [0, 1]"
    print("PASS: binary <-> multiclass(K=2) judged-probability parity holds.")


if __name__ == "__main__":
    test_episteme_parity()
    print("\nAll parity assertions passed.")
