# GPax

A codebase for Gaussian processes in Jax.

Disclaimer: This is not an officially supported Google product.

## Gaussian Process Probes (GPP)

GPP provides a method to measure uncertainty (aleatoric and epistemic) in linear probes trained on frozen embeddings. This library supports both binary and multiclass classification settings.

Please find algorithm descriptions in *[Gaussian Process Probes (GPP) for Uncertainty-Aware Probing](https://arxiv.org/abs/2305.18213)*.

### 1. Multiclass Classification (New)

For problems with more than two classes ($K > 2$), use the `gpp_multiclass` interface.

#### Mathematical Explanation

To extend GPP to multiclass classification, we model the probability vector $\mathbf{p}$ using a **Dirichlet distribution** instead of the Beta distribution used in the binary case.

1.  **Latent Function**: We model $K$ independent latent functions $f_1, \dots, f_K$ using Gaussian Processes.
2.  **Dirichlet Approximation**: The Dirichlet distribution parameters $\boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_K]$ are linked to the latent functions via a Log-Normal approximation. Specifically, we approximate the Dirichlet distribution by mapping the GP outputs to the concentration parameters:
    $$ \ln \alpha_k \approx f_k $$
    This allows us to perform standard GP inference for each class independently while preserving the properties of the Dirichlet prior.
3.  **Likelihood**: The observed labels are treated as Categorical draws from the probability vector $\mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha})$.

#### Usage

```python
from GPax.probing.probabilistic_probe_multiclass import gpp_multiclass

# x_train: (n_train, d) - Training embeddings
# y_train: (n_train,) or (n_train, K) - Integer or One-hot labels
# x_query: (n_query, d) - Test embeddings to probe

measures = gpp_multiclass(
    x_query=x_query,
    x_observed=x_train,
    y_observed=y_train,
    num_classes=10,       # Optional if y_observed is one-hot
    alpha_eps=0.1,        # Prior concentration (lower = sparser)
    strength=5.0,         # Observation strength
    n=10000,              # Monte Carlo samples
    seed=42
)

print(measures['information_gain'])  # Epistemic uncertainty
print(measures['expected_aleatory_entropy']) # Aleatoric uncertainty
```

### 2. Binary Classification (Original)

For binary tasks, you can continue to use the standard `gpp` function which utilizes a Beta prior.

#### Mathematical Explanation

In the binary case, we model the probability $p$ of the positive class using a **Beta distribution**.
1.  **Latent Function**: A single latent function $f$ models the log-odds or similar transformation.
2.  **Beta Approximation**: The Beta parameters $(\alpha, \beta)$ are approximated using a Log-Normal distribution to enable tractable GP inference.
3.  **Likelihood**: Observed labels are Bernoulli distributed.

#### Usage

To use binary GPP, call [`gpp`](https://github.com/google-research/gpax/blob/main/GPax/probing/probabilistic_probe.py#L25):

```python
from GPax.probing.probabilistic_probe import gpp

# y_train must be 0 or 1
measures = gpp(
    x_query=x_query,
    x_observed=x_train,
    y_observed=y_train,
    alpha_eps=0.1,
    strength=5.0
)
```

### 3. Baselines

The library includes standard probabilistic probing baselines adapted for both binary and multiclass settings.

#### Linear Probe Ensemble (LPE)
-   **Math**: Uses bootstrap aggregation (bagging) of $M$ Logistic Regression models. Epistemic uncertainty is measured by the mutual information of the ensemble predictions:
    $$ I(y; \theta | x) = H(\mathbb{E}[p(y|x, \theta)]) - \mathbb{E}[H(p(y|x, \theta))] $$
-   **Usage**: `lpe` (binary) or `lpe_multiclass` (multiclass).

#### Maximum Probability (MaxProb)
-   **Math**: Uses a single deterministic Logistic Regression model. The proxy for epistemic uncertainty is the maximum predicted probability:
    $$ \text{score} = \max_k p(y=k|x) $$
    Lower MaxProb implies higher uncertainty (often used for OOD detection).
-   **Usage**: `lp_maxprob` (binary) or `lp_maxprob_multiclass` (multiclass).

#### Mahalanobis Distance
-   **Math**: Models the features of each class $k$ as a Gaussian $\mathcal{N}(\boldsymbol{\mu}_k, \boldsymbol{\Sigma})$. The OOD score is the negative minimum Mahalanobis distance to any class centroid:
    $$ \text{score} = -\min_k (x - \boldsymbol{\mu}_k)^T \boldsymbol{\Sigma}^{-1} (x - \boldsymbol{\mu}_k) $$
-   **Usage**: `maha` (binary) or `maha_multiclass` (multiclass).

## Pre-trained Gaussian processes

Please find algorithm descriptions in *[Pre-trained Gaussian processes for Bayesian optimization](https://arxiv.org/abs/2109.08215)*. An alternative implementation can be found at https://github.com/google-research/hyperbo.

Implemented models include vanilla Gaussian processes ([`GaussianProcess`](https://github.com/google-research/gpax/blob/main/GPax/models/gp.py#L74)) as well as meta and multi-task Gaussian processes ([`MultiTaskGaussianProcess`](https://github.com/google-research/gpax/blob/main/GPax/models/gp.py#L279)).

For pre-training the multi-task Gaussian process, you can call an optimizer (minimization) on the [empirical KL divergence (EKL) objective](https://github.com/google-research/gpax/blob/main/GPax/objectives/empirical_kl_divergence.py) or the [negative log likelihood (NLL) objective](https://github.com/google-research/gpax/blob/main/GPax/objectives/neg_log_likelihood.py). Examples of evaluating these objectives can be found in [the test for EKL](https://github.com/google-research/gpax/blob/main/GPax/objectives/empirical_kl_divergence_test.py) and [the test for NLL](https://github.com/google-research/gpax/blob/main/GPax/objectives/neg_log_likelihood_test.py).

We also implemented [classic acquisition functions](https://github.com/google-research/gpax/blob/main/GPax/bayesopt/acquisitions.py) for Bayesian optimization. See [`GPax/bayesopt/acquisitions_test.py`](https://github.com/google-research/gpax/blob/main/GPax/bayesopt/acquisitions_test.py) for an example of how to evaluate these acquisition functions.

### Citation

```
@article{wang2023gpp,
  title={{Gaussian Process Probes (GPP) for Uncertainty-Aware Probing}},
  author={Zi Wang and
          Alexander Ku and
          Jason Baldridge and
          Thomas L Griffiths and
          Been Kim},
  journal={arXiv preprint arXiv:2305.18213},
  year={2023}
}
```

```
@article{wang2023hyperbo,
  title={{Pre-trained Gaussian processes for Bayesian optimization}},
  author={Zi Wang and
          George E. Dahl and
          Kevin Swersky and
          Chansoo Lee and
          Zachary Nado and
          Justin Gilmer and
          Jasper Snoek and
          Zoubin Ghahramani},
  journal={arXiv preprint arXiv:2109.08215},
  year={2023}
}
```

## Installation

### Conda / Mamba (Recommended)

To set up the development environment using Conda or Mamba:

```bash
# Create the environment from the yaml file
mamba env create -f environment.yaml

# Activate the environment
mamba activate gpax-multiclass
```

### Pip / Venv

We recommend using Python 3.10 or higher.

To install the latest development version inside a virtual environment, run
```bash
python3 -m venv env-pd
source env-pd/bin/activate
pip install --upgrade pip
pip install -e .
```

### Legacy Install (Git)
To install directly from git:
```bash
pip install "git+https://github.com/google-research/gpax.git#egg=gpax"
```
