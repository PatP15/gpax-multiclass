# coding=utf-8
"""Measure uncertainty for GPP, GPR, linear ensemble in PyTorch."""

import torch
import numpy as np
import sklearn.linear_model as sklm
from GPtorch.probing import gp

def gpp(
    x_query,
    x_observed=None,
    y_observed=None,
    alpha_eps=0.1,
    strength=5.0,
    n=int(1e5),
    seed=0,
):
  """Probing model representations with GPP."""
  # Convert inputs to tensors if needed (gp functions handle it, but good practice)
  # If y_observed is (N,), reshape to (N, 1)
  if y_observed is not None:
      if isinstance(y_observed, np.ndarray): y_observed = torch.tensor(y_observed)
      if y_observed.ndim == 1:
          y_observed = y_observed.unsqueeze(1)
          
  mean_func = gp.constant_mean
  cov_func = gp.cosine_kernel
  
  # Ensure params are tensors (though retrieve_params handles floats if we implemented it to)
  # In GPtorch/probing/gp.py we used retrieve_params which expects dictionary values.
  # If we pass floats, they might be used as is or converted.
  # Let's ensure they are consistent with x_query device/dtype if possible.
  
  # We'll let the internal functions handle conversion or pass python floats where acceptable.
  params = {
      'alpha_eps': torch.tensor(alpha_eps),
      'strength': torch.tensor(strength),
  }
  
  if isinstance(x_query, torch.Tensor) and x_query.is_cuda:
      params = {k: v.to(x_query.device) for k, v in params.items()}
      
  predictions = gp.beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed,
      params=params,
  )
  
  measures = gp.beta_gp_uncertainty(predictions, seed=seed, n=n)
  
  # Flatten/extract the first column (as in JAX version `x[:, 0]`)
  # measures values are (N, 1) or similar.
  return {k: v.squeeze() for k, v in measures.items()}


def gpr(x_query, x_observed=None, y_observed=None):
  """GP regression for classification with a default params."""
  if y_observed is not None:
      if isinstance(y_observed, np.ndarray): y_observed = torch.tensor(y_observed)
      if y_observed.ndim == 1:
          y_observed = y_observed.unsqueeze(1)

  mean_func = gp.constant_mean
  cov_func = gp.cosine_kernel
  params = {
      'constant': 0.0,
      'noise_variance': 0.1,
      'scale': 6.0,
      'signal_variance': 6.0,
  }
  
  # Convert params to tensors
  device = x_query.device if isinstance(x_query, torch.Tensor) else torch.device('cpu')
  params = {k: torch.tensor(v, device=device) for k, v in params.items()}
  
  mu, var = gp.gpr_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed,
      params=params,
  )
  measures = gp.gp_uncertainty(mu, var)
  return {k: v.squeeze() for k, v in measures.items()}


def lpe(x_query, x_observed=None, y_observed=None, repeats=int(1e2)):
  """Linear probe ensemble using bootstrap."""
  # Use sklearn, so convert to numpy
  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed
  
  # Ensure y_observed is 1D for sklm
  y_observed_np = y_observed_np.flatten()
  
  p_samples = []
  pos_idx = np.where(y_observed_np)[0]
  neg_idx = np.where(y_observed_np == 0)[0]
  
  if pos_idx.shape[0] == 0 or neg_idx.shape[0] == 0:
    raise ValueError('Must have at least 1 positive and 1 negative examples.')
    
  n_obs = len(x_observed_np) - 2
  
  for _ in range(repeats):
    idx_0 = np.random.choice(pos_idx)
    idx_1 = np.random.choice(neg_idx)
    idx = np.random.choice(np.arange(n_obs + 2), (n_obs,))
    idx = np.hstack(([idx_0, idx_1], idx))
    
    cls = sklm.LogisticRegression().fit(x_observed_np[idx], y_observed_np[idx])
    cls_p = cls.predict_proba(x_query_np)[:, 1]
    p_samples.append(cls_p)
    
  p_samples = np.array(p_samples).T  # num_inputs x n
  
  # Use PyTorch implementation of uncertainty metrics
  # Convert back to tensor
  p_samples_torch = torch.tensor(p_samples, device=x_query.device if isinstance(x_query, torch.Tensor) else 'cpu')
  
  measures = gp.classifier_samples_uncertainty(p_samples_torch, True)
  return {k: v.squeeze() for k, v in measures.items()}


def lp_maxprob(x_query, x_observed=None, y_observed=None):
  """Maximum predicted probability for OOD detection."""
  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed
  
  y_observed_np = y_observed_np.flatten()

  cls = sklm.LogisticRegression().fit(x_observed_np, y_observed_np)
  cls_p = cls.predict_proba(x_query_np)[:, 1]
  
  # np.max([cls_p, 1-cls_p], axis=0)
  res = np.max([cls_p, 1 - cls_p], axis=0)
  
  if isinstance(x_query, torch.Tensor):
      res = torch.tensor(res, device=x_query.device)
      
  return {'episteme': res}


def maha(x_query, x_observed=None, y_observed=None):
  """Mahalanobis score."""
  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed
  
  y_observed_np = y_observed_np.flatten()

  x0 = x_observed_np[np.where(y_observed_np == 0)[0]]
  x1 = x_observed_np[np.where(y_observed_np)[0]]
  
  if x0.shape[0] == 0 or x1.shape[0] == 0:
    raise ValueError('Must have at least 1 positive and 1 negative examples.')
    
  if x0.shape[0] == 1 and x1.shape[0] == 1:
    res = np.zeros(len(x_query_np))
    if isinstance(x_query, torch.Tensor): res = torch.tensor(res, device=x_query.device)
    return {'episteme': res}
    
  mu0 = np.mean(x0, axis=0)
  cov0 = np.cov(x0.T, bias=True)
  mu1 = np.mean(x1, axis=0)
  cov1 = np.cov(x1.T, bias=True)
  
  cov = cov0 + cov1
  if not cov.shape:
    cov = np.array([[cov]])
    
  cov_inv = np.linalg.pinv(cov)
  
  delta0 = x_query_np - mu0
  delta1 = x_query_np - mu1
  
  dist0 = np.sum(np.dot(delta0, cov_inv) * delta0, axis=1)
  dist1 = np.sum(np.dot(delta1, cov_inv) * delta1, axis=1)
  
  res = -np.min([dist0, dist1], axis=0)
  
  if isinstance(x_query, torch.Tensor):
      res = torch.tensor(res, device=x_query.device)
      
  return {'episteme': res}


def svm_probe(x_query, x_observed=None, y_observed=None):
  """SVM probe baseline."""
  import sklearn.svm as sksvm
  
  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed
  
  y_observed_np = y_observed_np.flatten()

  cls = sksvm.SVC(kernel='linear', probability=True).fit(x_observed_np, y_observed_np)
  probs = cls.predict_proba(x_query_np)[:, 1]
  
  episteme = np.max([probs, 1 - probs], axis=0)
  preds = cls.predict(x_query_np)
  
  if isinstance(x_query, torch.Tensor):
      probs = torch.tensor(probs, device=x_query.device)
      episteme = torch.tensor(episteme, device=x_query.device)
      preds = torch.tensor(preds, device=x_query.device)
  
  return {
      'Judged probability': probs,
      'episteme': episteme,
      'predictions': preds
  }

