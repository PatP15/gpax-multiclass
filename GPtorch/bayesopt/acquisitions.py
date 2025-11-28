# coding=utf-8
"""Acquisition functions for Bayesian optimization in PyTorch."""

import abc
import enum
import torch
import numpy as np

class Goal(enum.Enum):
  MAXIMIZE = 'MAXIMIZE'
  MINIMIZE = 'MINIMIZE'


class AcquisitionFunction(abc.ABC):

  @abc.abstractmethod
  def evaluate(self, posterior) -> torch.Tensor:
    """Evaluates the acquisition function."""
    pass


class Quantile(AcquisitionFunction):
  """Take the p-th quantile of the posterior."""

  def __init__(self, quantile):
    self.quantile = torch.tensor(quantile) if not torch.is_tensor(quantile) else quantile

  def evaluate(self, posterior) -> torch.Tensor:
    # posterior must support icdf (inverse cdf) or quantile
    # torch.distributions has icdf
    return posterior.icdf(self.quantile)


class UpperConfidenceBound(AcquisitionFunction):
  """UCB Acquisition function."""

  def __init__(self, beta):
    self.beta = torch.tensor(beta) if not torch.is_tensor(beta) else beta

  def evaluate(self, posterior) -> torch.Tensor:
    return posterior.mean + posterior.stddev * self.beta


class ImprovementZScore(AcquisitionFunction):
  """Computes the Z score of the improvement."""

  def __init__(self, target, goal: Goal):
    self.target = torch.tensor(target) if not torch.is_tensor(target) else target
    self.goal = goal

  def evaluate(self, posterior) -> torch.Tensor:
    multiplier = 1.0 if self.goal == Goal.MAXIMIZE else -1.0
    return multiplier * (self.target - posterior.mean) / posterior.stddev


class ExpectedImprovement(AcquisitionFunction):
  """Expected Improvement."""

  def __init__(self, target, goal: Goal):
    self.target = torch.tensor(target) if not torch.is_tensor(target) else target
    self.goal = goal

  def evaluate(self, posterior) -> torch.Tensor:
    # Using ImprovementZScore
    z_score_fn = ImprovementZScore(self.target, self.goal)
    gamma = z_score_fn.evaluate(posterior)
    
    normal = torch.distributions.Normal(0., 1.)
    if gamma.device != torch.device('cpu'):
        # Need normal on same device?
        # torch distributions usually create tensors on cpu by default if params are scalars?
        # But if we pass tensors, it matches.
        # Here 0. and 1. are floats.
        # Gamma has device.
        # We can just use normal.log_prob(gamma).exp() ?
        # Or construct normal with tensors on device.
        normal = torch.distributions.Normal(
            torch.tensor(0., device=gamma.device), 
            torch.tensor(1., device=gamma.device)
        )
        
    return (normal.log_prob(gamma).exp() + gamma * normal.cdf(gamma)) * posterior.stddev 
    # Wait, JAX code: (normal.prob(gamma) - gamma * (1 - normal.cdf(gamma)))
    # Wait, let's check formula.
    # EI = sigma * (gamma * cdf(gamma) + pdf(gamma)) usually?
    # JAX code: `normal.prob(gamma) - gamma * (1 - normal.cdf(gamma))`
    # If gamma is (f_best - mean)/sigma (for minimization?)
    # JAX ImprovementZScore: multiplier * (target - mean) / stddev.
    # If Goal is MINIMIZE, multiplier is -1. target = f_best.
    # gamma = (mean - f_best) / sigma. (Standard Z score for minimization? No usually (f_best - mean))
    # If Goal is MAXIMIZE, multiplier is 1. target = f_best.
    # gamma = (f_best - mean) / sigma.
    
    # Standard EI (Maximization):
    # I = max(f - f_best, 0).
    # EI = (mu - f_best) * CDF(Z) + sigma * PDF(Z)
    # where Z = (mu - f_best) / sigma.
    
    # JAX ZScore: (f_best - mu) / sigma. So Z_jax = -Z_std.
    # JAX Formula: PDF(Z_jax) - Z_jax * (1 - CDF(Z_jax)).
    # PDF(-x) = PDF(x).
    # 1 - CDF(-x) = CDF(x).
    # So PDF(x) + x * CDF(x). This matches standard EI.
    
    # So we just port the JAX formula exactly.
    return (normal.log_prob(gamma).exp() - gamma * (1 - normal.cdf(gamma))) * posterior.stddev


class ThomsonSampling(AcquisitionFunction):
  """Samples from posterior."""

  def __init__(self, seed):
    self.seed = seed

  def evaluate(self, posterior) -> torch.Tensor:
    # seed handling for torch
    # If seed is int, make generator
    if isinstance(self.seed, int):
        gen = torch.Generator(device=posterior.mean.device)
        gen.manual_seed(self.seed)
        return posterior.sample(generator=gen)
    return posterior.sample()

