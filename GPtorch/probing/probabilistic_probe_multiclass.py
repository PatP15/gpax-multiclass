# coding=utf-8
"""Measure uncertainty for Multiclass GPP in PyTorch."""

import torch
import numpy as np
import sklearn.linear_model as sklm
from GPtorch.probing import gp_multiclass as gp
from GPtorch.probing import gp as gp_base

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
  """Probing model representations with GPP for multiclass classification."""
  mean_func = gp_base.constant_mean
  cov_func = gp_base.cosine_kernel
  
  params = {
      'alpha_eps': torch.tensor(alpha_eps),
      'strength': torch.tensor(strength),
  }
  
  if num_classes is not None:
      params['num_classes'] = torch.tensor(num_classes)
      
  if isinstance(x_query, torch.Tensor) and x_query.is_cuda:
      params = {k: v.to(x_query.device) for k, v in params.items()}
      
  predictions = gp.dirichlet_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed,
      params=params,
  )
  
  measures = gp.dirichlet_gp_uncertainty(predictions, seed=seed, n=n)
  return measures


def lpe_multiclass(x_query, x_observed=None, y_observed=None, repeats=int(1e2), num_classes=None):
  """Linear probe ensemble using bootstrap for multiclass classification."""
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  # Convert to numpy for sklearn
  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed

  # Convert one-hot to indices for sklearn if necessary
  if y_observed_np.ndim > 1 and y_observed_np.shape[1] > 1:
      y_indices = np.argmax(y_observed_np, axis=1)
      if num_classes is None:
          num_classes = y_observed_np.shape[1]
  else:
      y_indices = y_observed_np.flatten()
      if num_classes is None:
          num_classes = int(np.max(y_indices) + 1)
          
  p_samples = []
  n_obs = len(x_observed_np)
  classes = np.unique(y_indices)
  
  if len(classes) < 2:
       raise ValueError("Must have at least 2 classes in the training data.")

  for _ in range(repeats):
    idx = np.random.choice(np.arange(n_obs), (n_obs,))
    
    cls = sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
    try:
        cls.fit(x_observed_np[idx], y_indices[idx])
    except Exception:
        continue
        
    cls_p = cls.predict_proba(x_query_np) # n_query x n_classes_in_bootstrap
    
    if cls_p.shape[1] != num_classes:
        full_p = np.zeros((x_query_np.shape[0], num_classes))
        for i, c in enumerate(cls.classes_):
            if c < num_classes:
                full_p[:, int(c)] = cls_p[:, i]
        p_samples.append(full_p)
    else:
        p_samples.append(cls_p)

  if not p_samples:
      raise RuntimeError("Failed to fit any bootstrap models.")

  p_samples = np.array(p_samples).transpose(1, 2, 0)  # num_inputs x num_classes x repeats
  
  # Convert to torch
  p_samples_torch = torch.tensor(p_samples, device=x_query.device if isinstance(x_query, torch.Tensor) else 'cpu')
  
  measures = gp.classifier_samples_uncertainty_multiclass(p_samples_torch)
  return measures


def lp_maxprob_multiclass(x_query, x_observed=None, y_observed=None):
  """Maximum predicted probability for OOD detection (multiclass)."""
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed

  if y_observed_np.ndim > 1 and y_observed_np.shape[1] > 1:
      y_indices = np.argmax(y_observed_np, axis=1)
  else:
      y_indices = y_observed_np.flatten()

  cls = sklm.LogisticRegression(multi_class='multinomial', solver='lbfgs', max_iter=1000)
  cls.fit(x_observed_np, y_indices)
  cls_p = cls.predict_proba(x_query_np)
  
  max_p = np.max(cls_p, axis=1)
  
  res = max_p
  if isinstance(x_query, torch.Tensor):
      res = torch.tensor(res, device=x_query.device)
      
  return {'episteme': res}


def maha_multiclass(x_query, x_observed=None, y_observed=None, num_classes=None):
  """Mahalanobis score for multiclass."""
  if x_observed is None or y_observed is None:
      raise ValueError("x_observed and y_observed must be provided.")

  if isinstance(x_query, torch.Tensor): x_query_np = x_query.cpu().numpy()
  else: x_query_np = x_query
      
  if isinstance(x_observed, torch.Tensor): x_observed_np = x_observed.cpu().numpy()
  else: x_observed_np = x_observed
      
  if isinstance(y_observed, torch.Tensor): y_observed_np = y_observed.cpu().numpy()
  else: y_observed_np = y_observed

  if y_observed_np.ndim > 1 and y_observed_np.shape[1] > 1:
      y_indices = np.argmax(y_observed_np, axis=1)
      if num_classes is None:
          num_classes = y_observed_np.shape[1]
  else:
      y_indices = y_observed_np.flatten().astype(int)
      if num_classes is None:
          num_classes = int(np.max(y_indices) + 1)

  unique_classes = np.unique(y_indices)
  
  means = []
  
  # pooled_cov
  pooled_cov = np.zeros((x_observed_np.shape[1], x_observed_np.shape[1]))
  
  for c in unique_classes:
      xc = x_observed_np[y_indices == c]
      if xc.shape[0] == 0:
          continue
          
      mu_c = np.mean(xc, axis=0)
      means.append(mu_c)
      
      if xc.shape[0] > 1:
          cov_c = np.cov(xc.T, bias=True)
          if cov_c.ndim == 0:
             cov_c = np.array([[cov_c]])
      else:
          cov_c = np.zeros((x_observed_np.shape[1], x_observed_np.shape[1]))
          
      pooled_cov += cov_c
      
  if len(means) == 0:
       res = np.zeros(len(x_query_np))
       if isinstance(x_query, torch.Tensor): res = torch.tensor(res, device=x_query.device)
       return {'episteme': res}
       
  inv_cov = np.linalg.pinv(pooled_cov)
  
  dists = []
  for mu_c in means:
      delta = x_query_np - mu_c
      dist = np.sum(np.dot(delta, inv_cov) * delta, axis=1)
      dists.append(dist)
      
  dists = np.array(dists) 
  min_dist = np.min(dists, axis=0)
  
  res = -min_dist
  if isinstance(x_query, torch.Tensor):
      res = torch.tensor(res, device=x_query.device)
      
  return {'episteme': res}

