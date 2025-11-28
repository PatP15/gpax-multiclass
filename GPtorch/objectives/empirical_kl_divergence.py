# coding=utf-8
"""Empirical KL divergence in PyTorch."""

import logging
import torch
import numpy as np
from GPtorch import utils

def inverse_spdmatrix_vector_product(spd_matrix, x):
  """Computes the inverse matrix vector product where the matrix is SPD."""
  L = torch.linalg.cholesky(spd_matrix)
  return torch.cholesky_solve(x, L)


def kl_multivariate_normal(mu0,
                           cov0,
                           mu1,
                           cov1,
                           weight=1.,
                           partial=True,
                           feat0=None,
                           eps=0.):
  """Computes KL divergence between two multivariate normal distributions."""
  # Ensure inputs are tensors
  if not isinstance(cov0, torch.Tensor): cov0 = torch.tensor(cov0)
  if not isinstance(cov1, torch.Tensor): cov1 = torch.tensor(cov1)
  if not isinstance(mu0, torch.Tensor): mu0 = torch.tensor(mu0)
  if not isinstance(mu1, torch.Tensor): mu1 = torch.tensor(mu1)
  
  # Handle scalar covariances
  if cov0.ndim == 0: cov0 = cov0.view(1, 1)
  if cov1.ndim == 0: cov1 = cov1.view(1, 1)
  
  if eps > 0.:
    eye0 = torch.eye(cov0.shape[0], device=cov0.device, dtype=cov0.dtype)
    eye1 = torch.eye(cov1.shape[0], device=cov1.device, dtype=cov1.dtype)
    cov0 = cov0 + eye0 * eps
    cov1 = cov1 + eye1 * eps

  mu_diff = mu1 - mu0
  
  # cov1 inverse product
  chol1 = torch.linalg.cholesky(cov1)
  
  # inv(cov1) @ mu_diff
  # We need column vector for mu_diff if it's (N,)
  if mu_diff.ndim == 1: mu_diff = mu_diff.unsqueeze(1)
  
  cov1invmudiff = torch.cholesky_solve(mu_diff, chol1)
  
  # trace(inv(cov1) @ cov0)
  # = trace(solve(cov1, cov0))
  # = sum(diag(solve(cov1, cov0)))
  
  cov1invcov0 = torch.cholesky_solve(cov0, chol1)
  trcov1invcov0 = torch.trace(cov1invcov0)
  
  # mahalanobis = mu_diff.T @ inv(cov1) @ mu_diff
  mahalanobis = torch.matmul(mu_diff.T, cov1invmudiff)
  
  # logdet
  logdetcov1 = 2 * torch.sum(torch.log(torch.diag(chol1)))
  
  common_terms = trcov1invcov0 + mahalanobis + logdetcov1
  
  if partial:
    return 0.5 * weight * common_terms
  else:
    if feat0 is not None and feat0.shape[0] > feat0.shape[1]:
      logging.info('Using pseudo determinant of cov0.')
      # slogdet
      feat_prod = torch.matmul(feat0.T, feat0) / feat0.shape[1]
      sign, logdetcov0 = torch.linalg.slogdet(feat_prod)
      
      logging.info(msg=f'Pseudo logdetcov0 = {logdetcov0}')
      # assert sign == 1.
      
      cov0inv = torch.linalg.pinv(cov0)
      rank = torch.linalg.matrix_rank(torch.matmul(cov0inv, cov0))
      
      term = common_terms - logdetcov0 - rank + np.log(2 * np.pi) * (cov1.shape[0] - feat0.shape[1])
      return 0.5 * weight * term
    else:
      sign, logdetcov0 = torch.linalg.slogdet(cov0)
      logging.info(msg=f'sign = {sign}; logdetcov0 = {logdetcov0}')
      # assert sign == 1.
      return 0.5 * weight * (common_terms - logdetcov0 - cov0.shape[0])


def objective(model,
              params,
              dataset,
              partial: bool = True):
  """Compute empirical KL divergence of model to empirical estimates."""
  
  kl_sum = 0.0
  
  for sub_dataset in dataset:
      if sub_dataset.aligned is None:
          continue
          
      if sub_dataset.y.shape[0] == 0:
          continue
          
      mu_data = torch.mean(sub_dataset.y, dim=1)
      # cov expects (N, M) where rows are variables (if rowvar=True default in numpy)
      # torch.cov added in recent versions.
      # If not available, we calculate manually.
      # sub_dataset.y is (N, M). N features, M samples?
      # JAX: mean(axis=1) implies y is (N, M).
      # cov(y) -> (N, N).
      
      # torch.cov(input, correction=1, fweights=None, aweights=None)
      # input: (M, N) if we want cov of N variables? No, torch.cov matches numpy.
      # Expects variables as rows.
      cov_data = torch.cov(sub_dataset.y) # bias=False default (1/(N-1)). JAX used bias=True (1/N).
      # We should check if we need bias=True equivalent (correction=0).
      cov_data = torch.cov(sub_dataset.y, correction=0)
      
      # Prediction
      if params is not None and hasattr(torch, 'func'):
          from torch.func import functional_call
          pred = functional_call(model, params, (sub_dataset.x,))
      else:
          pred = model(sub_dataset.x)
          
      # pred should have .mean and .covariance or .variance methods/properties
      # torch.distributions.Distribution has .mean and .covariance_matrix (for MVN)
      
      # Assuming pred is MVN-like
      mu1 = pred.mean
      if hasattr(pred, 'covariance_matrix'):
          cov1 = pred.covariance_matrix
      else:
          # If it only has variance (diagonal)
          cov1 = torch.diag_embed(pred.variance)
          
      # feat0 = y - mu
      # y: (N, M). mu_data: (N,). mu_data[:, None]: (N, 1).
      feat0 = sub_dataset.y - mu_data.unsqueeze(1)
      
      kl_val = kl_multivariate_normal(
        mu0=mu_data,
        cov0=cov_data,
        mu1=mu1,
        cov1=cov1,
        partial=partial,
        feat0=feat0
      )
      
      kl_sum += kl_val
      
  return kl_sum

