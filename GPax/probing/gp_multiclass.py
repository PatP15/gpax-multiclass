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

"""Implementations of Gaussian processes for multiclass classification using Dirichlet distribution."""
import functools
import jax
import jax.numpy as jnp
import jax.scipy as jsp
import jax.scipy.linalg as jspla

vmap = jax.vmap


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


def constant_mean(params, x, warp_func=None):
  """Constant mean function."""
  (val,) = retrieve_params(params, ['constant'], warp_func)
  return jnp.full((x.shape[0], 1), val)


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
    cov_func = functools.partial(kernel, params, warp_func=warp_func)
    mmap = vmap(lambda x: vmap(lambda y: cov_func(x, y))(vx1))
    if vx2 is None:
      if diag:
        return vmap(lambda x: cov_func(x, x))(vx1)
      vx2 = vx1
    return mmap(vx2).T

  return matrix_map


@covariance_matrix
def squared_exponential_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r2 = jnp.sum(((x1 - x2) / lengthscale)**2)
  return jnp.squeeze(signal_variance) * jnp.exp(-r2 / 2)


@covariance_matrix
def laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r1 = jnp.sum((jnp.abs(x1 - x2) / lengthscale))
  return jnp.squeeze(signal_variance) * jnp.exp(-r1)


@covariance_matrix
def additive_laplace_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel: Eq.(4.9/13) of GPML book."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  r1 = jnp.abs(x1 - x2) / lengthscale
  return jnp.squeeze(signal_variance) * jnp.sum(jnp.exp(-r1))


@covariance_matrix
def squared_exponential_sphere_kernel(params, x1, x2, warp_func=None):
  """Squared exponential kernel on sphere distance."""
  params_keys = ['lengthscale', 'signal_variance']
  lengthscale, signal_variance = retrieve_params(params, params_keys, warp_func)
  cosine = get_cosine(params, x1, x2)
  cosine = jnp.clip(cosine, 0.0, 1.0)
  r = jnp.arccos(cosine)
  r2 = (r / lengthscale)**2
  return jnp.squeeze(signal_variance) * jnp.exp(-r2 / 2)


@covariance_matrix
def dot_product_kernel(params, x1, x2, warp_func=None):
  r"""Dot product kernel with normalized inputs."""
  params_keys = ['dot_prod_sigma', 'dot_prod_bias']
  sigma, bias = retrieve_params(params, params_keys, warp_func)
  return jnp.dot(x1, x2.T) / jnp.square(sigma) + jnp.square(bias)


def get_cosine_phi(params, vx, warp_func=None):
  """Get linear bases for Bayesian linear regression equivalent model."""
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
  signal_variance, = retrieve_params(params, ['signal_variance'], warp_func)
  delta = intercept_scaling**2
  r = jnp.sqrt(jnp.sum(vx**2, axis=1, keepdims=True) + delta)
  if intercept_scaling != 0:
    phi = jnp.hstack((vx, jnp.ones((len(vx), 1))*intercept_scaling)) / r
  else:
    phi = vx / r
  return phi * jnp.sqrt(signal_variance)


def get_cosine(params, x1, x2, warp_func=None):
  """Compute cosine between two vectors."""
  if 'intercept_scaling' in params:
    (intercept_scaling,) = retrieve_params(
        params, ['intercept_scaling'], warp_func
    )
  else:
    intercept_scaling = 1.0
  delta = intercept_scaling**2
  r1 = jnp.sqrt(jnp.sum(x1**2) + delta)
  r2 = jnp.sqrt(jnp.sum(x2**2) + delta)
  return (jnp.dot(x1, x2.T) + delta) / r1 / r2


@covariance_matrix
def cosine_kernel(params, x1, x2, warp_func=None):
  r"""Kernel defined by cosine similarity."""
  params_keys = ['signal_variance']
  signal_variance, = retrieve_params(params, params_keys, warp_func)
  return get_cosine(params, x1, x2, warp_func) * signal_variance


def get_latent_var_mu_dirichlet(alpha):
  """Get latent variance and mean for the lognormal distribution in Dirichlet GP.
  
  Args:
    alpha: n x K array of Dirichlet concentration parameters
  
  Returns:
    var: n x K array of log-normal variances
    mu: n x K array of log-normal means
  """
  var = jnp.log(1 / alpha + 1)
  mu = jnp.log(alpha) - var / 2
  return var, mu


def set_default_params_dirichlet(params, num_classes, warp_func=None):
  alpha_eps, = retrieve_params(
      params, ['alpha_eps'], warp_func=warp_func
  )
  params['signal_variance'] = jnp.log(1/alpha_eps + 1)
  y, _ = get_latent_observations_dirichlet(params, jnp.zeros((1, num_classes)), warp_func=warp_func)
  params['constant'] = jnp.min(y)  # (jnp.max(y) + jnp.min(y)) / 2
  return params


def get_latent_observations_dirichlet(params, y, warp_func=None):
  """Get latent observations of Dirichlet GP.

  Args:
    params: dictionary mapping from parameter keys to values.
    y: labels. One-hot encoded n x K or class indices n x 1.
    warp_func: optional dictionary that specifies the warping function for each
      parameter.

  Returns:
    y: latent function observation n x K.
    var: latent noise n x K.
  """
  
  alpha_eps, strength = retrieve_params(
      params, ['alpha_eps', 'strength'], warp_func=warp_func
  )
  
  # Check if y needs one-hot encoding
  if y.ndim == 1 or y.shape[1] == 1:
    if 'num_classes' in params:
        num_classes = params['num_classes']
    else:
        num_classes = int(jnp.max(y) + 1)
    y = jax.nn.one_hot(y.flatten().astype(int), num_classes)
    
  # Dirichlet parameters: α_k = α_eps + y_k × strength for each class k
  alpha = jnp.ones_like(y) * alpha_eps + y * strength
  
  var, y = get_latent_var_mu_dirichlet(alpha)
  return y, var


def gp_predict(
    *,
    mean_func,
    cov_func,
    params,
    x_query,  # n' x d array
    x_observed=None,  # n x d array or None
    y_observed=None,  # n x 1 array or None
    var_observed=None,  # flat array of size n or None
    warp_func=None,
    var_only=True,
    predict_weight=False,
):
  """Predict GP posterior with observed heteroscedastic noise variance."""
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
    return mu_query, cov_query[:, None]
  mu_observed = mean_func(params, x_observed, warp_func=warp_func)
  cov_observed = cov_func(params, x_observed, warp_func=warp_func) + jnp.diag(
      var_observed.flatten()
  )
  chol = jspla.cholesky(cov_observed, lower=True)
  delta_y = y_observed - mu_observed
  kinvy = jspla.cho_solve((chol, True), delta_y)
  cov_observed_query = cov_func(
      params, x_observed, x_query, warp_func=warp_func
  )
  mu = jnp.dot(cov_observed_query.T, kinvy) + mu_query
  v = jspla.solve_triangular(chol, cov_observed_query, lower=True)
  if var_only:
    diagdot = jax.vmap(lambda x: jnp.dot(x, x.T))
    var = cov_query - diagdot(v.T)
    cov = var[:, None]
  else:
    cov = cov_query - jnp.dot(v.T, v)
  if predict_weight:
    phi = get_cosine_phi(params, x_observed, warp_func=warp_func)  # n x d
    u = jnp.dot(phi.T, kinvy)
    sig = jspla.solve_triangular(chol, phi, lower=True)
    sig = jnp.diag(jnp.ones((phi.shape[1],))) - jnp.dot(sig.T, sig)
    return mu, cov, u, sig
  return mu, cov


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
  """Predict Dirichlet GP posterior for the latent function.

  Args:
    mean_func: mean function handle.
    cov_func: covariance function handle.
    params: dictionary mapping from parameter keys to values.
    x_query: n' x d input array to be queried.
    x_observed: observed n x d input array.
    y_observed: observed n x K one-hot or n x 1 class indices.
    warp_func: optional dictionary that specifies the warping function.
    var_only: return variance only if True.

  Returns:
    Predictions: a list of tuples. Each tuple is the posterior mean (n' x 1) and
    (co)variance (n' x n' or n' x 1) for the K functions.
  """
  if y_observed is None or y_observed.shape[0] == 0:
    y_latent, var_latent = None, None
    if 'num_classes' in params:
        num_classes = params['num_classes']
    else:
        # Default fallback if no obs and no num_classes specified
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
      
    mu_var = gp_predict(
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
  """Extract latent GPs for K classes.
  
  Returns:
    mus: list of K arrays (n' x 1) of means
    vars: list of K arrays (n' x 1) or (n' x n') of var/cov
  """
  mus = [pred[0] for pred in predictions]
  vars = [pred[1] for pred in predictions]
  return mus, vars


def get_softmax_quantiles(mus, vars, q):
  """Get quantiles for softmax(latent GPs).
  
  Args:
    mus: list of K arrays (n' x 1)
    vars: list of K arrays (n' x 1)
    q: quantile level
  """
  # Simplified version: just return the softmax of the means as point estimate
  mu_stacked = jnp.hstack(mus)
  # For a proper quantile we would need sampling, similar to uncertainty
  return jax.nn.softmax(mu_stacked, axis=1), None


def get_dirichlet_quantiles(predictions, q):
  """Get the median and (q, 1-q) quantiles of Dirichlet GP classification model."""
  mus, vars = get_latent_gp_dirichlet(predictions)
  return get_softmax_quantiles(mus, vars, q)


def dirichlet_gp_uncertainty(predictions, seed=0, n=int(1e6)):
  """Get measures of aleatory and epistemic uncertainties for a batch of queries.

  Args:
    predictions: Predictions returned by dirichlet_gp_predict.
    seed: int random seed.
    n: number of samples for Monte Carlo estimation.

  Returns:
    Dictionary mapping from name to measure.
  """
  mus, vars = get_latent_gp_dirichlet(predictions)
  
  # Stack to n' x K
  latent_mu = jnp.hstack(mus)  # n' x K
  
  # Handle diagonal vs full covariance
  latent_vars_list = []
  for v in vars:
    if v.shape[0] == v.shape[1] and v.ndim > 1:
       # Full covariance - extract diagonal
       latent_vars_list.append(jnp.diag(v)[:, None])
    else:
       latent_vars_list.append(v)
       
  latent_var = jnp.hstack(latent_vars_list)  # n' x K
  latent_var = jnp.maximum(latent_var, 1e-32)
  
  
  return gp_uncertainty_multiclass(latent_mu, latent_var, seed=seed, n=n)


def gp_uncertainty_multiclass(latent_mu, latent_var, seed=0, n=int(1e6)):
  """Measure uncertainty metrics for softmax GP.
  
  Args:
    latent_mu: n' x K array
    latent_var: n' x K array (diagonal variances)
  """
  key = jax.random.PRNGKey(seed)
  num_queries, num_classes = latent_mu.shape
  
  # Sample: (n' x K x n)
  norm_samples = jax.random.normal(key, (num_queries, num_classes, n))
  iid_samples = norm_samples * jnp.sqrt(latent_var)[:, :, None] + latent_mu[:, :, None]
  
  # Apply softmax along class dimension (axis 1)
  p_samples = jax.nn.softmax(iid_samples, axis=1)  # n' x K x n
  
  
  # Categorical uncertainty metrics
  ret = classifier_samples_uncertainty_multiclass(p_samples)
  
  ret.update({
      'latent_var': latent_var,
      'latent_mu': latent_mu,
  })
  return ret


def classifier_samples_uncertainty_multiclass(p_samples):
  """Measure uncertainty for categorical classifier samples.
  
  Args:
    p_samples: n' x K x n array of probability samples
  
  Returns:
    Dictionary with uncertainty measures.
  """
  eps = 1e-10
  
  # Mean prediction per class: p_bar_k = E[p_k]
  mu = jnp.mean(p_samples, axis=2)  # n' x K
  
  # Epistemic variance (variance in predictions per class)
  var = jnp.mean(p_samples**2, axis=2) - mu**2  # n' x K
  
  # Expected aleatoric entropy: E[H(p)] = E[ - sum p_k log p_k ]
  # Compute entropy for each sample first: H(p)_s = - sum_k p_ks log p_ks
  sample_entropies = -jnp.sum(p_samples * jnp.log(p_samples + eps), axis=1) # n' x n
  categorical_entropy = jnp.mean(sample_entropies, axis=1, keepdims=True) # n' x 1
  
  # Total uncertainty: H(E[p]) = - sum_k mu_k log mu_k
  entropy_of_mean = -jnp.sum(mu * jnp.log(mu + eps), axis=1, keepdims=True) # n' x 1
  
  # Mutual information (epistemic uncertainty) = Total - Aleatoric
  info_gain = entropy_of_mean - categorical_entropy
  
  
  # Episteme: Negative Entropy of the distribution of p (approximated as Gaussian)
  # To match binary GPP, higher Episteme means higher certainty (lower variance).
  # We use the sum of marginal entropies as an upper bound approximation for entropy of p.
  var_safe = jnp.maximum(var, 1e-32)
  approx_entropy = 0.5 * jnp.sum(jnp.log(2 * jnp.pi * jnp.e * var_safe), axis=1, keepdims=True)
  
  return {
      'epistemic_var': var,  # n' x K
      'categorical_mu': mu,  # n' x K
      'expected_aleatory_entropy': categorical_entropy,  # n' x 1
      'information_gain': info_gain,  # n' x 1
      'Alea': categorical_entropy,
      'Episteme': -approx_entropy, # Higher means more concentrated p
      'Judged probability': mu,
  }


def dirichlet_gp_ood_score(measures):
  """OOD score for a Dirichlet GP = negative summed latent posterior variance.

  Following the paper (§4.4), GPP uses the negative latent variance of the
  posterior latent function as a proxy for episteme for OOD detection: a high
  score (low posterior variance) means in-distribution, a low score means OOD.
  We sum the per-class posterior variances already exposed as `latent_var` by
  `dirichlet_gp_uncertainty` / `gpp_multiclass`, so this is the single source of
  truth for the score (no separate diag-extraction that could drift).

  Args:
    measures: the dict returned by `gpp_multiclass` / `dirichlet_gp_uncertainty`
      (must contain 'latent_var', shape n' x K).

  Returns:
    score: (n',) array; higher = more in-distribution.
  """
  return -jnp.sum(measures['latent_var'], axis=1)


def dirichlet_mnll(
    mean_func,
    cov_func,
    params,
    x_query,
    y_query,
    x_train,  # n x d
    y_train,  # n x K
    warp_func=None,
    seed=0,
    n_mc=int(1e4),
):
  """Mean negative log predictive likelihood of held-out queries.

  Uses the same Monte-Carlo softmax posterior predictive (E[softmax(f)]) as the
  rest of the module, so it is consistent with `categorical_mu`. This replaces
  an earlier version that reconstructed alpha from the posterior variance, which
  was inconsistent with the predictive used everywhere else.
  """
  if len(y_train.shape) == 1 or y_train.shape[1] == 1:
    y_train_flat = y_train.flatten().astype(int)
    if 'num_classes' in params:
      num_classes = params['num_classes']
    else:
      num_classes = int(jnp.max(y_train_flat) + 1)
    y_train = jax.nn.one_hot(y_train_flat, num_classes)
  else:
    num_classes = y_train.shape[1]

  if len(y_query.shape) == 1 or y_query.shape[1] == 1:
    y_query = jax.nn.one_hot(y_query.flatten().astype(int), num_classes)

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
  # Canonical MC-softmax posterior predictive (matches gp_uncertainty_multiclass).
  unc = dirichlet_gp_uncertainty(predictions, seed=seed, n=n_mc)
  p_pred = unc['categorical_mu']  # n' x K
  nll = -jnp.mean(jnp.sum(y_query * jnp.log(p_pred + 1e-10), axis=1))
  return nll


def dirichlet_gp_nll(
    mean_func,
    cov_func,
    params,
    x_train,  # n x d
    y_train,  # n x K
    warp_func=None,
):
  """Negative log *marginal* likelihood of the latent Dirichlet-GP regression.

  Summed over the K independent latent GPs. This is the standard GP NLML for
  each latent regression,

      0.5 rᵀ Kₙ⁻¹ r + Σ log diag(L) + 0.5 n log 2π,

  with Kₙ = K + diag(v_latent_k), r = y_latent_k − mean, and L the Cholesky
  factor of Kₙ. Because it requires no held-out labels, it can be used for
  label-free model selection (e.g. choosing the kernel lengthscale by minimizing
  this score). Returns a JAX scalar.
  """
  if len(y_train.shape) == 1 or y_train.shape[1] == 1:
    y_train_flat = y_train.flatten().astype(int)
    if 'num_classes' in params:
      num_classes = params['num_classes']
    else:
      num_classes = int(jnp.max(y_train_flat) + 1)
    y_train = jax.nn.one_hot(y_train_flat, num_classes)
  else:
    num_classes = y_train.shape[1]

  # Set the constant mean and signal_variance so the Beta-GP prior holds. Work on
  # a copy so we never mutate the caller's params dict (set_default_* writes in place).
  params = set_default_params_dirichlet(dict(params), num_classes, warp_func=warp_func)

  y_latent, var_latent = get_latent_observations_dirichlet(
      params, y_train, warp_func=warp_func
  )  # n x K, n x K
  cov = cov_func(params, x_train, warp_func=warp_func)  # n x n
  mean = mean_func(params, x_train, warp_func=warp_func)  # n x 1
  n = y_latent.shape[0]

  total = 0.0
  for k in range(num_classes):
    cov_noisy = cov + jnp.diag(var_latent[:, k])
    chol = jspla.cholesky(cov_noisy, lower=True)
    residual = y_latent[:, k:k + 1] - mean
    alpha = jspla.cho_solve((chol, True), residual)
    total += (
        0.5 * jnp.squeeze(residual.T @ alpha)
        + jnp.sum(jnp.log(jnp.diag(chol)))
        + 0.5 * n * jnp.log(2 * jnp.pi)
    )
  return total
