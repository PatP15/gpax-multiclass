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

