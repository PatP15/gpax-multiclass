# coding=utf-8
"""Implementations of Gaussian processes for classification in PyTorch."""
import functools
import torch
import numpy as np

# PyTorch doesn't have a direct equivalent to jax.vmap that acts exactly the same on all functions without care
# We will implement vectorization via broadcasting in the kernels + wrapper
# or using torch.vmap if available (PyTorch 2.0+).
# For maximum compatibility and "normal pytorch", broadcasting is preferred.

def _verify_params(model_params, expected_keys):
  """Verify that dictionary params has the expected keys."""
  if not set(expected_keys).issubset(set(model_params.keys())):
    # Sort keys for consistent error message
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


def constant_mean(params, x, warp_func=None):
  """Constant mean function."""
  (val,) = retrieve_params(params, ['constant'], warp_func)
  # x is n x d
  return torch.full((x.shape[0], 1), val.item(), device=x.device, dtype=x.dtype)


def covariance_matrix(kernel):
  """Decorator to kernels to obtain the covariance matrix."""

  @functools.wraps(kernel)
  def matrix_map(params, vx1, vx2=None, warp_func=None, diag=False):
    """Returns the kernel matrix of input array vx1 and input array vx2.

    Args:
      params: parameters for the kernel.
      vx1: n1 x d dimensional input array representing n1 data points.
      vx2: n2 x d dimensional input array representing n2 data points. If it is
        not specified, vx2 is set to be the same as vx1.
      warp_func: optional dictionary that specifies the warping function for
        each parameter.
      diag: flag for returning diagonal terms of the matrix (True) or the full
        matrix (False).

    Returns:
      The n1 x n2 dimensional covariance matrix derived from kernel evaluations
        on every pair of inputs from vx1 and vx2 by default. If diag=True and
        vx2=None, it returns the diagonal terms of the n1 x n1 covariance
        matrix.
    """
    # Ensure inputs are tensors
    if not isinstance(vx1, torch.Tensor):
        vx1 = torch.tensor(vx1)
    
    if vx2 is None:
      if diag:
        # Diagonal elements only: k(x, x) for each x in vx1
        # We can pass x1 directly if kernel supports element-wise operation
        # or we need to loop/vmap.
        # Our kernels will be written to support broadcasting.
        # For diag, we want output (N,).
        # Kernel(x1, x1) with broadcasting usually gives (N, N).
        # But if we pass x1 and x1 (N, D), and kernel operations are elementwise,
        # (x1-x1) is 0.
        # We need to be careful.
        # Let's assume kernels can handle (N, D) and (N, D) to produce (N,) if we don't unsqueeze.
        return kernel(params, vx1, vx1, warp_func=warp_func)
      vx2 = vx1

    if not isinstance(vx2, torch.Tensor):
        vx2 = torch.tensor(vx2)

    # Full matrix calculation
    # Reshape for broadcasting:
    # vx1: (N1, 1, D)
    # vx2: (1, N2, D)
    x1_expanded = vx1.unsqueeze(1)
    x2_expanded = vx2.unsqueeze(0)
    
    return kernel(params, x1_expanded, x2_expanded, warp_func=warp_func)

  return matrix_map


@covariance_matrix
def squared_exponential_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  
  # Ensure parameters are on the correct device/dtype if they are tensors
  # x1, x2 are tensors.
  
  # (x1 - x2) shape depends on input.
  # If broadcasted: (N1, N2, D).
  # If diag: (N, D).
  # lengthscale might be (D,) or scalar.
  
  diff = (x1 - x2) / lengthscale
  r2 = torch.sum(diff**2, dim=-1)
  
  return torch.squeeze(signal_variance) * torch.exp(-r2 / 2)


@covariance_matrix
def laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  
  diff = torch.abs(x1 - x2) / lengthscale
  r1 = torch.sum(diff, dim=-1)
  
  return torch.squeeze(signal_variance) * torch.exp(-r1)


@covariance_matrix
def additive_laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  
  r1 = torch.abs(x1 - x2) / lengthscale
  # sum(exp(-r1))
  return torch.squeeze(signal_variance) * torch.sum(torch.exp(-r1), dim=-1)


@covariance_matrix
def squared_exponential_sphere_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel on sphere distance."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  
  cosine = get_cosine(params, x1, x2)
  cosine = torch.clamp(cosine, 0.0, 1.0)
  r = torch.acos(cosine)
  r2 = (r / lengthscale)**2
  
  return torch.squeeze(signal_variance) * torch.exp(-r2 / 2)


@covariance_matrix
def dot_product_kernel(params, x1, x2, warp_func=None):
  r"""Dot product kernel with normalized inputs."""
  params_keys = ['dot_prod_sigma', 'dot_prod_bias']
  sigma, bias = retrieve_params(params, params_keys, warp_func)
  
  # If x1 is (N1, 1, D) and x2 is (1, N2, D)
  # x1 * x2 -> (N1, N2, D). Sum over D.
  dot_prod = torch.sum(x1 * x2, dim=-1)
  
  return dot_prod / torch.square(sigma) + torch.square(bias)


def get_cosine_phi(params, vx, warp_func=None):
  """Get linear bases for Bayesian linear regression equivalent model."""
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
  signal_variance, = retrieve_params(params, ['signal_variance'], warp_func)
  
  # Ensure scaling is tensor
  if not torch.is_tensor(intercept_scaling):
      intercept_scaling = torch.tensor(intercept_scaling, device=vx.device, dtype=vx.dtype)
      
  delta = intercept_scaling**2
  r = torch.sqrt(torch.sum(vx**2, dim=1, keepdim=True) + delta)
  
  if intercept_scaling != 0:
    ones = torch.ones((len(vx), 1), device=vx.device, dtype=vx.dtype) * intercept_scaling
    phi = torch.hstack((vx, ones)) / r
  else:
    phi = vx / r
    
  return phi * torch.sqrt(signal_variance)


def get_cosine(params, x1, x2, warp_func=None):
  """Compute cosine between two vectors."""
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
    
  if not torch.is_tensor(intercept_scaling):
      intercept_scaling = torch.tensor(intercept_scaling, device=x1.device, dtype=x1.dtype)
      
  delta = intercept_scaling**2
  
  # Note: x1, x2 might be broadcasted (N1, 1, D) and (1, N2, D)
  # sum(x**2) needs to be over dim=-1
  r1 = torch.sqrt(torch.sum(x1**2, dim=-1) + delta)
  r2 = torch.sqrt(torch.sum(x2**2, dim=-1) + delta)
  
  # Dot product
  dot = torch.sum(x1 * x2, dim=-1)
  
  return (dot + delta) / r1 / r2


@covariance_matrix
def cosine_kernel(params, x1, x2, warp_func=None):
  r"""Kernel defined by cosine similarity."""
  params_keys = ['signal_variance']
  signal_variance, = retrieve_params(params, params_keys, warp_func)
  return get_cosine(params, x1, x2, warp_func) * signal_variance


def get_latent_var_mu(alpha):
  """Get latent variance and mean for the lognormal distribution in Beta GP."""
  var = torch.log(1 / alpha + 1)
  mu = torch.log(alpha) - var / 2
  return var, mu


def set_default_params(params, warp_func=None):
  # Copy params to avoid side effects
  params = params.copy()
  (alpha_eps,) = retrieve_params(
      params, ['alpha_eps'], warp_func=warp_func
  )
  params['signal_variance'] = torch.log(1/alpha_eps + 1)
  
  # Call get_latent_observations with a dummy input
  # We need to know dtype/device. alpha_eps is a tensor.
  y_dummy = torch.zeros((1, 1), device=alpha_eps.device, dtype=alpha_eps.dtype)
  
  y, _ = get_latent_observations(params, y_dummy, warp_func=warp_func)
  params['constant'] = torch.min(y)
  return params


def get_latent_observations(params, y, warp_func=None):
  """Get latent observations of Beta GP."""
  alpha_eps, strength = retrieve_params(
      params, ['alpha_eps', 'strength'], warp_func=warp_func
  )
  
  # y: (N, 1).
  # We need alpha: (N, 2).
  # jnp.hstack([y, 1-y])
  
  # Ensure y is tensor
  if not torch.is_tensor(y):
      y = torch.tensor(y, device=alpha_eps.device, dtype=alpha_eps.dtype)
      
  stacked_y = torch.cat([y, 1 - y], dim=1)
  
  alpha = torch.ones((y.shape[0], 2), device=y.device, dtype=y.dtype) * alpha_eps + stacked_y * strength
  
  var, y_latent = get_latent_var_mu(alpha)
  return y_latent, var


def gp_predict(
    *,
    mean_func,
    cov_func,
    params,
    x_query,
    x_observed=None,
    y_observed=None,
    var_observed=None,
    warp_func=None,
    var_only=True,
    predict_weight=False,
):
  """Predict GP posterior with observed heteroscedastic noise variance."""
  # Convert inputs to tensors if needed
  if not isinstance(x_query, torch.Tensor): x_query = torch.tensor(x_query)
  
  mu_query = mean_func(params, x_query, warp_func=warp_func)
  cov_query = cov_func(params, x_query, warp_func=warp_func, diag=var_only)
  
  if (
      x_observed is None
      or x_observed.shape[0] == 0
      or y_observed is None
      or y_observed.shape[0] == 0
      or var_observed is None
      or var_observed.shape[0] == 0
  ):
    # If cov_query is (N,), return (N, 1) covariance?
    # JAX returns cov_query[:, None].
    # If cov_query is already (N, 1) (from our dot product implementation?), then (N, 1, 1)?
    # If var_only=True, cov_query is (N,). We want (N, 1).
    if var_only and cov_query.ndim == 1:
        return mu_query, cov_query.unsqueeze(1)
    return mu_query, cov_query

  if not isinstance(x_observed, torch.Tensor): x_observed = torch.tensor(x_observed)
  if not isinstance(y_observed, torch.Tensor): y_observed = torch.tensor(y_observed)
  if not isinstance(var_observed, torch.Tensor): var_observed = torch.tensor(var_observed)

  mu_observed = mean_func(params, x_observed, warp_func=warp_func)
  
  cov_observed = cov_func(params, x_observed, warp_func=warp_func) 
  # Add diagonal noise
  # torch.diag expects 1D. flatten var_observed.
  cov_observed = cov_observed + torch.diag(var_observed.reshape(-1))
  
  # Cholesky
  # torch.linalg.cholesky returns lower triangular
  L = torch.linalg.cholesky(cov_observed)
  
  delta_y = y_observed - mu_observed
  # kinvy = inv(K) @ delta_y = inv(L @ L.T) @ delta_y
  # torch.cholesky_solve(B, L) computes inv(L @ L.T) @ B
  kinvy = torch.cholesky_solve(delta_y, L)
  
  cov_observed_query = cov_func(
      params, x_observed, x_query, warp_func=warp_func
  )
  
  # mu = cov_observed_query.T @ kinvy + mu_query
  mu = torch.matmul(cov_observed_query.T, kinvy) + mu_query
  
  # v = solve(L, cov_observed_query)
  # solve_triangular(A, B, upper=False) -> solve AX=B for X.
  v = torch.linalg.solve_triangular(L, cov_observed_query, upper=False)
  
  if var_only:
    # var = cov_query - diag(v.T @ v)
    # v is (N_obs, N_query).
    # v.T @ v -> (N_query, N_query). Diagonal is sum(v**2, axis=0).
    
    v_squared_sum = torch.sum(v**2, dim=0)
    var = cov_query - v_squared_sum
    # cov = var[:, None]
    cov = var.unsqueeze(1)
  else:
    cov = cov_query - torch.matmul(v.T, v)
    
  if predict_weight:
    phi = get_cosine_phi(params, x_observed, warp_func=warp_func)  # n x d
    u = torch.matmul(phi.T, kinvy)
    sig = torch.linalg.solve_triangular(L, phi, upper=False)
    sig = torch.diag(torch.ones((phi.shape[1],), device=phi.device)) - torch.matmul(sig.T, sig)
    return mu, cov, u, sig
    
  return mu, cov


def beta_gp_predict(
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
  """Predict Beta GP posterior for the latent function."""
  if y_observed is None or y_observed.shape[0] == 0:
    y_latent, var_latent = None, None
  else:
    y_latent, var_latent = get_latent_observations(
        params, y_observed, warp_func=warp_func
    )  # n x 2, n x 2
    
  # hacky way of setting constant mean
  params = set_default_params(params, warp_func=warp_func)
  predictions = []
  for i in range(2):
    if y_latent is not None:
      y_obs = y_latent[:, i:i+1]
      var_obs = var_latent[:, i]
    else:
      y_obs, var_obs = None, None
      
    mu_var = gp_predict(
        mean_func=mean_func,
        cov_func=cov_func,
        params=params,
        x_query=x_query,
        x_observed=x_observed,
        y_observed=y_obs,
        var_observed=var_obs,
        warp_func=warp_func,
        var_only=var_only,
    )
    predictions.append(mu_var)
  return predictions


def get_logistic_quantiles(mu, var, q):
  """Get the median and (q, 1-q) quantiles for logistic transform."""
  y = 1 / (1 + torch.exp(-mu))
  
  # norm.ppf
  normal = torch.distributions.Normal(mu, torch.sqrt(var))
  
  # q needs to be tensor for broadcasting if multiple qs?
  # JAX code passed [q, 1-q].
  if isinstance(q, float):
      qs = torch.tensor([q, 1-q], device=mu.device, dtype=mu.dtype)
  else:
      qs = q
      
  # icdf expects probabilities.
  # If mu is (N, 1). Normal shape is (N, 1).
  # icdf(qs) -> (N, 1) if qs is scalar?
  # We want (N, 2) output if qs has 2 elements.
  
  # We might need to expand qs to match batch shape if using broadcasting
  # Or loop.
  # torch.distributions broadcasting:
  # normal.icdf(value). value must be broadcastable to batch_shape.
  
  # If mu is (N, 1), batch_shape is (N, 1).
  # If qs is (2,), we can reshape to (2, 1, 1) or use iterating.
  
  q1 = normal.icdf(torch.tensor(q, device=mu.device))
  q2 = normal.icdf(torch.tensor(1-q, device=mu.device))
  
  quantiles = torch.stack([q1, q2], dim=-1) # (N, 1, 2) ?
  
  quantiles = 1 / (1 + torch.exp(-quantiles))
  return y, quantiles


def get_latent_gp(predictions):
  mu, var = (
      predictions[0][0] - predictions[1][0],  # delta of mu
      predictions[0][1] + predictions[1][1],  # sum of var or cov
  )
  return mu, var


def get_beta_quantiles(predictions, q):
  """Get the median and (q, 1-q) quantiles of Beta GP classification model."""
  mu, var = get_latent_gp(predictions)
  return get_logistic_quantiles(mu, var, q)


def get_mvn_samples(mu, cov, seed=0, n=int(1e6)):
  # Handle shapes
  if mu.shape[0] != cov.shape[0] or cov.shape[0] != cov.shape[1]:
    raise ValueError(
        f'mu.shape={mu.shape} and cov.shape={cov.shape}. Mean and cov shape'
        ' must match. Cov must be a square matrix.'
    )

  gen = torch.Generator(device=mu.device)
  gen.manual_seed(seed)
  
  # mu: (N, 1)
  # cov: (N, N)
  # samples: (N, n)
  
  norm_samples = torch.randn(mu.shape[0], n, device=mu.device, dtype=mu.dtype, generator=gen)
  L = torch.linalg.cholesky(cov)
  
  # samples = L @ norm_samples + mu
  samples = torch.matmul(L, norm_samples) + mu
  return samples, L


def beta_gp_uncertainty(predictions, seed=0, n=int(1e6)):
  """Get measures of aleatory and epistemic uncertainties for a batch of queries."""
  latent_mu, latent_var = get_latent_gp(predictions)
  if latent_var.shape[0] == latent_var.shape[1]:
    # predictions include covaraince instead of variance.
    latent_var = torch.diag(latent_var).unsqueeze(1)
  
  latent_var = torch.maximum(latent_var, torch.tensor(1e-32, device=latent_var.device))
  return gp_uncertainty(latent_mu, latent_var, seed=seed, n=n)


def gp_uncertainty(latent_mu, latent_var, seed=0, n=int(1e6)):
  """Measure uncertainty metrics for logistic GP."""
  gen = torch.Generator(device=latent_mu.device)
  gen.manual_seed(seed)
  
  norm_samples = torch.randn(latent_mu.shape[0], n, device=latent_mu.device, dtype=latent_mu.dtype, generator=gen)
  iid_samples = norm_samples * torch.sqrt(latent_var) + latent_mu
  p_samples = 1.0 / (1 + torch.exp(-iid_samples))  # num_inputs x n
  
  ret = classifier_samples_uncertainty(p_samples)
  
  norm_entropy = 0.5 * torch.log(2 * np.pi * latent_var) + 0.5
  
  # mean over axis 1 (samples)
  entropy = (
      norm_entropy
      + latent_mu
      - 2 * torch.mean(torch.log(1 + torch.exp(iid_samples)), dim=1, keepdim=True)
  )
  
  ret.update({
      'epistemic_entropy': entropy,
      'latent_var': latent_var,
      'latent_mu': latent_mu,
      'Episteme': -entropy,
  })
  return ret


def classifier_samples_uncertainty(
    p_samples,  # num_inputs x n
    approx_entropy=False,
):
  """Measure uncertainty metrics with classifier samples."""
  mu = torch.mean(p_samples, dim=1, keepdim=True)
  var = torch.mean(p_samples**2, dim=1, keepdim=True) - mu**2
  
  bernoulli_var = torch.mean(p_samples * (1 - p_samples), dim=1, keepdim=True)
  
  # Avoid log(0)
  eps = 1e-10
  term = p_samples * torch.log(p_samples + eps) + (1 - p_samples) * torch.log(1 - p_samples + eps)
  bernoulli_entropy = -torch.mean(term, dim=1, keepdim=True)
  
  info_gain = (
      -(mu * torch.log(mu + eps) + (1 - mu) * torch.log(1 - mu + eps)) - bernoulli_entropy
  )
  
  ret = {
      'epistemic_var': var,
      'bernoulli_mu': mu,
      'expected_aleatory_entropy': bernoulli_entropy,
      'expected_aleatory_var': bernoulli_var,
      'information_gain': info_gain,
      'Alea': bernoulli_entropy,
      'Judged probability': mu,
  }
  
  if approx_entropy:
    var = torch.maximum(var, torch.tensor(1e-32, device=var.device))
    entropy = 0.5 * torch.log(2 * np.pi * var) + 0.5
    ret['epistemic_entropy'] = entropy
    ret['Episteme'] = -entropy
    
  return ret


def estimate_entropy_from_mu_chol(mu, chol, samples):
  """Monte Carlo estimation of entropy for Beta GP using n samples."""
  if (
      mu.shape[0] != chol.shape[0]
      or chol.shape[0] != chol.shape[1]
      or samples.shape[0] != mu.shape[0]
  ):
    raise ValueError(
        f'mu.shape={mu.shape}, chol.shape={chol.shape},'
        f' samples.shape={samples.shape}. Mean and chol shape must match. chol'
        ' must be a square matrix.'
    )

  mvn_entropy = torch.sum(torch.log(torch.diag(chol))) + 0.5 * mu.shape[0] * (
      np.log(2 * np.pi) + 1.0
  )
  sum_mu = torch.sum(mu)
  additional_term = 2 * torch.sum(torch.nn.functional.softplus(samples)) / samples.shape[1]
  return mvn_entropy + sum_mu - additional_term


def estimate_batch_entropy(mu, cov, seed=0, n=int(1e6)):
  """Monte Carlo estimation of entropy for Beta GP using n samples."""
  samples, chol = get_mvn_samples(mu, cov, seed=seed, n=n)
  return estimate_entropy_from_mu_chol(mu, chol, samples)


def mvn_nll(y, mu, cov):
  """Negative log likelihood for one sample of Multivariate Normal."""
  if y.shape != mu.shape:
    raise ValueError('Shape of y and mu must match.')
  if y.shape[1] != 1 or mu.shape[1] != 1:
    raise ValueError('y and mu must be column vectors.')
  if y.shape[0] != cov.shape[0]:
    raise ValueError('Shape of y and cov must match.')
  if cov.shape[0] != cov.shape[1]:
    raise ValueError('Cov must be a square matrix.')
    
  y_diff = y - mu
  L = torch.linalg.cholesky(cov)
  # kinvy = solve(cov, y_diff)
  kinvy = torch.cholesky_solve(y_diff, L)
  
  return torch.sum(
      0.5 * torch.matmul(y_diff.T, kinvy)
      + torch.sum(torch.log(torch.diag(L)))
      + 0.5 * y.shape[0] * np.log(2 * np.pi)
  )


def beta_mnll(
    mean_func,
    cov_func,
    params,
    x_query,
    y_query,
    x_train,
    y_train,
    warp_func=None,
):
  """MNLL Eq 2 of Milios et al., 2018."""
  if len(y_train.shape) == 1:
    y_train = y_train.unsqueeze(1)
    
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_query,
      x_observed=x_train,
      y_observed=y_train,
      warp_func=warp_func,
      var_only=True,
  )
  
  alpha = 1 / (torch.exp(predictions[0][1]) - 1)
  beta = 1 / (torch.exp(predictions[1][1]) - 1)
  
  # y_query is boolean/0-1?
  # jnp.where(y_query, ...)
  mnll = torch.where(y_query.bool(), alpha / (alpha + beta), beta / (alpha + beta))
  return -torch.sum(torch.log(mnll))


def beta_gp_nll(
    mean_func,
    cov_func,
    params,
    x_train,
    y_train,
    warp_func=None,
):
  """Negative log data likelihood for Beta GP."""
  if len(y_train.shape) == 1:
    y_train = y_train.unsqueeze(1)
    
  params = set_default_params(params, warp_func=warp_func) # This returns copied params
  
  predictions = beta_gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_train,
      warp_func=warp_func,
      var_only=False,
  )

  # Loop over samples
  # vmap in JAX iterates over samples.
  # We can loop or assume batch.
  # mvn_nll takes (N_features, 1) vectors.
  # But here we have N samples.
  # beta_gp_predict(x_train) -> predictions for each sample?
  # predictions[0][0] is (N, 1) mean for class 0.
  # predictions[0][1] is (N, N) covariance for class 0.
  
  # The JAX code computes NLL treating all samples as ONE multivariate normal vector (size N)?
  # Wait, `vmap(vmap_func)(y_train.T).mean(axis=0)`
  # If y_train is (N, 1). y_train.T is (1, N).
  # vmap iterates over classes (if N classes)? No.
  # If y_train is (N, 1), vmap runs once?
  
  # In `beta_gp_nll` (JAX):
  # y_latent, var_latent = get_latent_observations(...)
  # y_latent is (N, 2).
  # Loop over 2 classes (i in range(y_latent.shape[1])).
  # mvn_nll(y_latent[:, i], mu, cov).
  # This computes the NLL of the entire vector of N samples.
  
  # So we don't loop over samples. We compute NLL of the (N,) vector.
  
  # So `vmap` in JAX might be confusingly used or I misinterpreted.
  # Ah, `vmap(vmap_func)(y_train.T)`.
  # If y_train is (N, 1). y_train.T is (1, N).
  # If vmap iterates over leading dim, it iterates 1 time.
  # Inside, `y` is (N,).
  
  # So yes, we just need to compute for the full batch.
  
  y_latent, var_latent = get_latent_observations(params, y_train, warp_func=warp_func)
  
  nll = 0
  for i in range(y_latent.shape[1]):
      var = var_latent[:, i]
      cov = predictions[i][1] + torch.diag(var.flatten())
      
      # y_latent[:, i:i+1] is (N, 1)
      nll += mvn_nll(y_latent[:, i:i+1], predictions[i][0], cov)
      
  # JAX returns .mean(axis=0). But nll list had 1 element?
  # vmap returned array of nlls.
  # If vmap ran once, it returned [nll]. mean is nll.
  
  return nll


def get_probit_quantiles(mu, var, q):
  """Get the median and (q, 1-q) quantiles for cumulative Gaussian transform."""
  normal = torch.distributions.Normal(0, 1)
  y = normal.cdf(mu.flatten())
  
  # quantiles
  # ppf([q, 1-q]) -> standard normal quantiles
  if isinstance(q, float):
      qs = torch.tensor([q, 1-q], device=mu.device, dtype=mu.dtype)
  else:
      qs = q
      
  z_scores = normal.icdf(qs) # (2,)
  
  # mu / sqrt(var+1) + z * sqrt(var) ?
  # JAX:
  # ppf([q, 1-q], loc=mu/sqrt(var+1), scale=sqrt(var))
  # = loc + z * scale
  
  loc = mu / torch.sqrt(var + 1)
  scale = torch.sqrt(var)
  
  # Broadcasting (N, 1) and (2,) -> (N, 2)
  loc = loc.squeeze() # (N,)
  scale = scale.squeeze() # (N,)
  
  q_vals = loc.unsqueeze(1) + scale.unsqueeze(1) * z_scores.unsqueeze(0)
  
  quantiles = normal.cdf(q_vals)
  return y, quantiles


def gpr_predict(
    mean_func,
    cov_func,
    params,
    x_query,
    x_observed=None,
    y_observed=None,
    warp_func=None,
    var_only=True,
):
  """Predict GP (classification as regression) posterior for the latent function."""
  noise_variance, scale = retrieve_params(
      params, ['noise_variance', 'scale'], warp_func=warp_func
  )
  y_observed = (y_observed * 2 - 1) * scale
  var_observed = torch.ones(x_observed.shape[0], device=x_observed.device, dtype=x_observed.dtype) * noise_variance
  
  return gp_predict(
      mean_func=mean_func,
      cov_func=cov_func,
      params=params,
      x_query=x_query,
      x_observed=x_observed,
      y_observed=y_observed,
      var_observed=var_observed,
      warp_func=warp_func,
      var_only=var_only,
  )
