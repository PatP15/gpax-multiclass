# coding=utf-8
"""Implementations of Gaussian processes for multiclass classification using Dirichlet distribution in PyTorch."""
import functools
import torch
import numpy as np
from GPtorch.probing import gp

def _verify_params(model_params, expected_keys):
  """Verify that dictionary params has the expected keys."""
  if not set(expected_keys).issubset(set(model_params.keys())):
    raise ValueError(
        f'Expected parameters are {sorted(expected_keys)}, '
        f'but received {sorted(model_params.keys())}.'
    )


def retrieve_params(params, keys, warp_func):
  """Returns a list of parameter values (warped if specified) by keys' order."""
  _verify_params(params, keys)
  if warp_func:
    values = [
        warp_func[key](params[key]) if key in warp_func else params[key]
        for key in keys
    ]
  else:
    values = [params[key] for key in keys]
  return values


def get_latent_var_mu_dirichlet(alpha):
  """Get latent variance and mean for the lognormal distribution in Dirichlet GP."""
  # alpha: n x K
  var = torch.log(1 / alpha + 1)
  mu = torch.log(alpha) - var / 2
  return var, mu


def get_latent_observations_dirichlet(params, y, warp_func=None):
  """Get latent observations of Dirichlet GP."""
  alpha_eps, strength = retrieve_params(
      params, ['alpha_eps', 'strength'], warp_func=warp_func
  )
  
  # Ensure y is tensor
  if not torch.is_tensor(y):
      # Assume y is numpy or list. If float/int, convert to tensor.
      # We need to know device/dtype from alpha_eps if possible.
      y = torch.tensor(y, device=alpha_eps.device) # dtype depends on content (int for indices, float for one-hot)

  # Check if y needs one-hot encoding
  if y.ndim == 1 or y.shape[1] == 1:
    if 'num_classes' in params:
        num_classes = int(params['num_classes'].item()) if torch.is_tensor(params['num_classes']) else params['num_classes']
    else:
        num_classes = int(torch.max(y).item() + 1)
    
    # y must be Long/Int for one_hot
    y_long = y.long().reshape(-1)
    y = torch.nn.functional.one_hot(y_long, num_classes).to(dtype=alpha_eps.dtype)
    
  # Dirichlet parameters: α_k = α_eps + y_k × strength for each class k
  alpha = torch.ones_like(y) * alpha_eps + y * strength
  
  var, y_latent = get_latent_var_mu_dirichlet(alpha)
  return y_latent, var


def set_default_params_dirichlet(params, num_classes, warp_func=None):
  # Copy params
  params = params.copy()
  (alpha_eps,) = retrieve_params(
      params, ['alpha_eps'], warp_func=warp_func
  )
  params['signal_variance'] = torch.log(1/alpha_eps + 1)
  
  # Dummy call
  y_dummy = torch.zeros((1, num_classes), device=alpha_eps.device, dtype=alpha_eps.dtype)
  y, _ = get_latent_observations_dirichlet(params, y_dummy, warp_func=warp_func)
  params['constant'] = torch.min(y)
  return params


def dirichlet_gp_predict(
    *,
    mean_func,
    cov_func,
    params,
    x_query,
    x_observed=None,
    y_observed=None,
    warp_func=None,
    var_only=True,
):
  """Predict Dirichlet GP posterior for the latent function."""
  if y_observed is None or y_observed.shape[0] == 0:
    y_latent, var_latent = None, None
    if 'num_classes' in params:
        num_classes = int(params['num_classes'].item()) if torch.is_tensor(params['num_classes']) else params['num_classes']
    else:
        num_classes = 2 
  else:
    y_latent, var_latent = get_latent_observations_dirichlet(
        params, y_observed, warp_func=warp_func
    )  # n x K, n x K
    num_classes = y_latent.shape[1]
    
  # hacky way of setting constant mean
  params = set_default_params_dirichlet(params, num_classes, warp_func=warp_func)
  
  predictions = []
  
  for i in range(num_classes):
    if y_latent is not None:
      y_observed_i = y_latent[:, i:i+1]
      var_observed_i = var_latent[:, i]
    else:
      y_observed_i, var_observed_i = None, None
      
    mu_var = gp.gp_predict(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x_query=x_query,
        x_observed=x_observed,
        y_observed=y_observed_i,
        var_observed=var_observed_i,
        warp_func=warp_func,
        var_only=var_only,
    )
    predictions.append(mu_var)
    
  return predictions


def get_latent_gp_dirichlet(predictions):
  """Extract latent GPs for K classes."""
  mus = [pred[0] for pred in predictions]
  vars = [pred[1] for pred in predictions]
  return mus, vars


def get_softmax_quantiles(mus, vars, q):
  """Get quantiles for softmax(latent GPs)."""
  # Simplified version: just return the softmax of the means as point estimate
  # mus is list of (N, 1).
  mu_stacked = torch.cat(mus, dim=1) # (N, K)
  return torch.nn.functional.softmax(mu_stacked, dim=1), None


def get_dirichlet_quantiles(predictions, q):
  """Get the median and (q, 1-q) quantiles of Dirichlet GP classification model."""
  mus, vars = get_latent_gp_dirichlet(predictions)
  return get_softmax_quantiles(mus, vars, q)


def dirichlet_gp_uncertainty(predictions, seed=0, n=int(1e6)):
  """Get measures of aleatory and epistemic uncertainties for a batch of queries."""
  mus, vars = get_latent_gp_dirichlet(predictions)
  
  # Stack to n' x K
  latent_mu = torch.cat(mus, dim=1)  # n' x K
  
  # Handle diagonal vs full covariance
  latent_vars_list = []
  for v in vars:
    if v.ndim > 1 and v.shape[0] == v.shape[1]:
       # Full covariance - extract diagonal
       latent_vars_list.append(torch.diag(v).unsqueeze(1))
    else:
       # Assuming v is already diagonal (N, 1) or (N,)
       if v.ndim == 1: v = v.unsqueeze(1)
       latent_vars_list.append(v)
       
  latent_var = torch.cat(latent_vars_list, dim=1)  # n' x K
  latent_var = torch.maximum(latent_var, torch.tensor(1e-32, device=latent_var.device))
  
  return gp_uncertainty_multiclass(latent_mu, latent_var, seed=seed, n=n)


def gp_uncertainty_multiclass(latent_mu, latent_var, seed=0, n=int(1e6)):
  """Measure uncertainty metrics for softmax GP."""
  gen = torch.Generator(device=latent_mu.device)
  gen.manual_seed(seed)
  
  num_queries, num_classes = latent_mu.shape
  
  # Sample: (n' x K x n)
  norm_samples = torch.randn(num_queries, num_classes, n, device=latent_mu.device, dtype=latent_mu.dtype, generator=gen)
  
  # latent_mu[:, :, None] -> (N, K, 1)
  # latent_var[:, :, None] -> (N, K, 1)
  iid_samples = norm_samples * torch.sqrt(latent_var).unsqueeze(2) + latent_mu.unsqueeze(2)
  
  # Apply softmax along class dimension (axis 1)
  p_samples = torch.nn.functional.softmax(iid_samples, dim=1)  # n' x K x n
  
  # Categorical uncertainty metrics
  ret = classifier_samples_uncertainty_multiclass(p_samples)
  
  ret.update({
      'latent_var': latent_var,
      'latent_mu': latent_mu,
  })
  return ret


def classifier_samples_uncertainty_multiclass(p_samples):
  """Measure uncertainty for categorical classifier samples."""
  eps = 1e-10
  
  # Mean prediction per class: p_bar_k = E[p_k]
  # Average over samples (dim 2)
  mu = torch.mean(p_samples, dim=2)  # n' x K
  
  # Epistemic variance (variance in predictions per class)
  var = torch.mean(p_samples**2, dim=2) - mu**2  # n' x K
  
  # Expected aleatoric entropy: E[H(p)] = E[ - sum p_k log p_k ]
  # Compute entropy for each sample first: H(p)_s = - sum_k p_ks log p_ks
  # Sum over classes (dim 1)
  sample_entropies = -torch.sum(p_samples * torch.log(p_samples + eps), dim=1) # n' x n
  categorical_entropy = torch.mean(sample_entropies, dim=1, keepdim=True) # n' x 1
  
  # Total uncertainty: H(E[p]) = - sum_k mu_k log mu_k
  # Sum over classes (dim 1)
  entropy_of_mean = -torch.sum(mu * torch.log(mu + eps), dim=1, keepdim=True) # n' x 1
  
  # Mutual information (epistemic uncertainty) = Total - Aleatoric
  info_gain = entropy_of_mean - categorical_entropy
  
  return {
      'epistemic_var': var,  # n' x K
      'categorical_mu': mu,  # n' x K
      'expected_aleatory_entropy': categorical_entropy,  # n' x 1
      'information_gain': info_gain,  # n' x 1
      'Alea': categorical_entropy,
      'Episteme': info_gain,
      'Judged probability': mu,
  }


def dirichlet_mnll(
    mean_func,
    cov_func,
    params,
    x_query,
    y_query,
    x_train,
    y_train,
    warp_func=None,
):
  """Mean negative log likelihood for Dirichlet GP."""
  # Handle input shapes
  if len(y_train.shape) == 1 or y_train.shape[1] == 1:
    y_train_flat = y_train.long().flatten()
    if 'num_classes' in params:
        num_classes = int(params['num_classes'].item()) if torch.is_tensor(params['num_classes']) else params['num_classes']
    else:
        num_classes = int(torch.max(y_train_flat).item() + 1)
    y_train = torch.nn.functional.one_hot(y_train_flat, num_classes).to(dtype=x_train.dtype)
  else:
    num_classes = y_train.shape[1]
  
  if len(y_query.shape) == 1 or y_query.shape[1] == 1:
    y_query_flat = y_query.long().flatten()
    y_query = torch.nn.functional.one_hot(y_query_flat, num_classes).to(dtype=x_train.dtype)
  
  predictions = dirichlet_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_query,
      x_observed=x_train,
      y_observed=y_train,
      warp_func=warp_func,
      var_only=True,
  )
  
  alphas = []
  for pred in predictions:
      alpha_k = 1 / (torch.exp(pred[1]) - 1)
      alphas.append(alpha_k)
      
  alpha_stacked = torch.cat(alphas, dim=1)  # n' x K
  alpha_sum = torch.sum(alpha_stacked, dim=1, keepdim=True)
  
  # Predicted probabilities
  p_pred = alpha_stacked / alpha_sum  # n' x K
  
  # Negative log likelihood: - sum( y_k * log(p_k) )
  # Sum over classes and samples? Usually mean over samples.
  # JAX code: -jnp.sum(y_query * jnp.log(p_pred + 1e-10))
  # This is total NLL (sum over all queries).
  nll = -torch.sum(y_query * torch.log(p_pred + 1e-10))
  return nll


def dirichlet_gp_nll(
    mean_func,
    cov_func,
    params,
    x_train,
    y_train,
    warp_func=None,
):
  """Negative log data likelihood for Dirichlet GP."""
  if len(y_train.shape) == 1 or y_train.shape[1] == 1:
    y_train_flat = y_train.long().flatten()
    if 'num_classes' in params:
        num_classes = int(params['num_classes'].item()) if torch.is_tensor(params['num_classes']) else params['num_classes']
    else:
        num_classes = int(torch.max(y_train_flat).item() + 1)
    y_train = torch.nn.functional.one_hot(y_train_flat, num_classes).to(dtype=x_train.dtype)
  else:
    num_classes = y_train.shape[1]

  # hacky way of setting constant mean
  params['constant'] = set_default_params_dirichlet(params, num_classes, warp_func=warp_func)['constant']
  
  predictions = dirichlet_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_train,
      warp_func=warp_func,
      var_only=False,
  )

  y_latent, var_latent = get_latent_observations_dirichlet(
        params, y_train, warp_func=warp_func
  )
  
  nll = 0
  for i in range(y_latent.shape[1]):
      var = var_latent[:, i]
      cov = predictions[i][1] + torch.diag(var.flatten())
      nll += gp.mvn_nll(y_latent[:, i:i+1], predictions[i][0], cov)
  
  # Return sum? JAX returned mean over samples but inside vmap it summed?
  # `return jnp.sum(jnp.array(nll))` inside vmap_func.
  # vmap_func is applied to each sample? No, we established vmap applied once to the whole batch (N).
  # So nll is scalar (sum over all samples).
  # Then `.mean(axis=0)` on a scalar/1-element array is just the value.
  
  return nll

