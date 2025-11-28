# coding=utf-8
# Copyright 2024 GPax Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Measure uncertainty for Multiclass GPP."""

from functools import partial
from GPax.probing import gp_multiclass as gp
import jax
import jax.numpy as jnp
import numpy as np
import sklearn.linear_model as sklm


@partial(jax.jit, static_argnames=['num_classes', 'n'])
def gpp_multiclass(
    x_query,
    x_observed=None,
    y_observed=None,
    num_classes=None,
    alpha_eps=0.1,
    strength=5.0,
    n=int(1e5),
    seed=0,
):
  """Probing model representations with GPP for multiclass classification.

  Details in the GPP paper: https://arxiv.org/pdf/2305.18213.pdf.

  Args:
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array. Set to None if no observations.
    y_observed: observed n x K one-hot OR n x 1 indices.
    num_classes: number of classes K (optional, inferred if None).
    alpha_eps: the episilon parameter in the Dirichlet prior Dir(alpha_eps, ...).
    strength: the s parameter in posterior inference.
    n: number of samples for Monte Carlo estimation.
    seed: int random seed for Monte Carlo estimation.

  Returns:
    Dictionary mapping from name to measures of uncertainty.
  """
  jax.debug.print("Starting GPP Multiclass...")
  jax.debug.print("Query shape: {}", x_query.shape)
  if x_observed is not None:
    jax.debug.print("Observed shape: {}, labels: {}", x_observed.shape, y_observed.shape)
    
  mean_func = gp.constant_mean
  cov_func = gp.cosine_kernel
  params = {
      'alpha_eps': alpha_eps,
      'strength': strength,
  }
  if num_classes is not None:
      params['num_classes'] = num_classes
      
  predictions = gp.dirichlet_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      # Ensure y_observed is passed correctly. 
      # If it is indices (n,), expand to (n, 1) for consistency with internal checks if needed,
      # though gp_multiclass handles both.
      y_observed=y_observed[:, None] if (y_observed is not None and y_observed.ndim == 1) else y_observed,
      params=params,
  )
  
  measures = gp.dirichlet_gp_uncertainty(predictions, seed=seed, n=n)
  
  jax.debug.print("Uncertainty dict keys: {}", list(measures.keys()))
  
  # Return full measures (n' x K or n' x 1)
  return measures


def lpe_multiclass(x_query, x_observed=None, y_observed=None, repeats=int(1e2), num_classes=None):
  """Linear probe ensemble using bootstrap for multiclass classification.
  
  Args:
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array.
    y_observed: observed n x 1 indices or n x K one-hot.
    repeats: number of ensemble members.
    num_classes: number of classes K (optional).
  
  Returns:
    Dictionary mapping from name to measures of uncertainty.
  """
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  # Convert one-hot to indices for sklearn if necessary
  if y_observed.ndim > 1 and y_observed.shape[1] > 1:
      y_indices = np.argmax(y_observed, axis=1)
      if num_classes is None:
          num_classes = y_observed.shape[1]
  else:
      y_indices = y_observed.flatten()
      if num_classes is None:
          num_classes = int(np.max(y_indices) + 1)
          
  p_samples = []
  n_obs = len(x_observed)
  classes = np.unique(y_indices)
  
  if len(classes) < 2:
       raise ValueError("Must have at least 2 classes in the training data.")

  for _ in range(repeats):
    # Bootstrap resampling
    # Ensure we get at least one sample from each present class to avoid errors
    # Simpler approach: just standard bootstrap, if a class is missing, 
    # sklearn handles it but might not output prob for that class. 
    # To serve as a robust baseline, we force at least one sample per class if possible,
    # or just accept standard bootstrap. Let's do standard bootstrap.
    idx = np.random.choice(np.arange(n_obs), (n_obs,))
    
    # Check if all classes are present, if not, retry or augment
    # For simplicity, we just fit on what we have. 
    # However, we need output shape to be (n_query, num_classes).
    
    # Use LogisticRegression with multinomial
    cls = sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    try:
        cls.fit(x_observed[idx], y_indices[idx])
    except Exception:
        # If fit fails (e.g. only 1 class selected), skip this member
        continue
        
    cls_p = cls.predict_proba(x_query) # n_query x n_classes_in_bootstrap
    
    # If some classes were missing in bootstrap, we need to pad
    if cls_p.shape[1] != num_classes:
        full_p = np.zeros((x_query.shape[0], num_classes))
        # map existing classes
        for i, c in enumerate(cls.classes_):
            if c < num_classes:
                full_p[:, int(c)] = cls_p[:, i]
        p_samples.append(full_p)
    else:
        p_samples.append(cls_p)

  if not p_samples:
      raise RuntimeError("Failed to fit any bootstrap models.")

  p_samples = np.array(p_samples).transpose(1, 2, 0)  # num_inputs x num_classes x repeats
  
  # Use JAX version of uncertainty calc
  # Convert to JAX array
  p_samples_jax = jnp.array(p_samples)
  
  measures = gp.classifier_samples_uncertainty_multiclass(p_samples_jax)
  return measures


def lp_maxprob_multiclass(x_query, x_observed=None, y_observed=None):
  """Maximum predicted probability for OOD detection (multiclass)."""
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  # Convert one-hot to indices
  if y_observed.ndim > 1 and y_observed.shape[1] > 1:
      y_indices = np.argmax(y_observed, axis=1)
  else:
      y_indices = y_observed.flatten()

  cls = sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
  cls.fit(x_observed, y_indices)
  cls_p = cls.predict_proba(x_query) # n_query x K
  
  # Max probability across classes
  max_p = np.max(cls_p, axis=1)
  
  # Epistemic uncertainty proxy: usually 1 - max_p or similar. 
  # The original code returned `np.max([cls_p, 1 - cls_p])` for binary.
  # Here we just return max_p as 'episteme' (higher = more confident, so less uncertain).
  # Or maybe we should return 1 - max_p to align with "uncertainty"?
  # Original `lp_maxprob` returns a value where HIGHER means ???
  # In original: `np.max([p, 1-p])` is confidence (0.5 to 1.0).
  # So 'episteme' there meant confidence.
  return {'episteme': max_p}


def maha_multiclass(x_query, x_observed=None, y_observed=None, num_classes=None):
  """Mahalanobis score for multiclass."""
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  # Convert one-hot to indices
  if y_observed.ndim > 1 and y_observed.shape[1] > 1:
      y_indices = np.argmax(y_observed, axis=1)
      if num_classes is None:
          num_classes = y_observed.shape[1]
  else:
      y_indices = y_observed.flatten().astype(int)
      if num_classes is None:
          num_classes = int(np.max(y_indices) + 1)

  unique_classes = np.unique(y_indices)
  
  means = []
  covs = []
  
  # Compute class means and pooled covariance
  pooled_cov = np.zeros((x_observed.shape[1], x_observed.shape[1]))
  
  for c in unique_classes:
      xc = x_observed[y_indices == c]
      if xc.shape[0] == 0:
          continue
          
      mu_c = np.mean(xc, axis=0)
      means.append(mu_c)
      
      if xc.shape[0] > 1:
          cov_c = np.cov(xc.T, bias=True)
          # Handle scalar covariance if d=1 (unlikely for embeddings but possible)
          if cov_c.ndim == 0:
             cov_c = np.array([[cov_c]])
      else:
          cov_c = np.zeros((x_observed.shape[1], x_observed.shape[1]))
          
      # Weighted by number of samples? Original code just summed cov0 + cov1
      # Let's sum them to match original logic
      pooled_cov += cov_c
      
  if len(means) == 0:
       return {'episteme': np.zeros(len(x_query))}
       
  # Invert pooled covariance
  # Add small noise for stability if needed? Original used pinv.
  inv_cov = np.linalg.pinv(pooled_cov)
  
  dists = []
  for mu_c in means:
      delta = x_query - mu_c
      # Mahalanobis dist: delta^T * inv_cov * delta
      # Vectorized: sum( (delta @ inv_cov) * delta, axis=1 )
      dist = np.sum(np.dot(delta, inv_cov) * delta, axis=1)
      dists.append(dist)
      
  dists = np.array(dists) # num_classes_present x n_query
  min_dist = np.min(dists, axis=0)
  
  # Return negative distance so that higher value = "closer" / "more confident"
  return {'episteme': -min_dist}
