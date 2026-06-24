# Dirichlet-GPP Multiclass Extension: Mathematical Derivation and Synthetic Verification

This document provides an exhaustive technical specification of the Gaussian Process Probing (GPP) extension from binary (Beta) to multiclass (Dirichlet) classification. It details the mathematical derivation, the exact code implementation, and the verification experiments performed on the **3DShapes synthetic dataset**.

**Note:** This documentation focuses exclusively on the core GPP-Dirichlet methodology and its verification on controlled synthetic data.

---

## 1. Mathematical Formulation

The goal is to model a multiclass classification problem with $K$ classes, where the target variable $y \in \{1, \dots, K\}$. We seek a probabilistic model that provides not just a class prediction, but a full distribution over the probability simplex $\Delta^{K-1}$, allowing for the decomposition of uncertainty into **aleatoric** (data noise) and **epistemic** (model ignorance) components.

### 1.1 The Dirichlet Prior
In the binary GPP, we modeled the probability $p$ of the positive class using a Beta distribution. For the multiclass case ($K > 2$), we generalize this to a **Dirichlet distribution**:

$$ \mathbf{p} \sim \text{Dir}(\boldsymbol{\alpha}) $$

where:
*   $\mathbf{p} = [p_1, \dots, p_K]^\top$ is the probability vector such that $\sum_{k=1}^K p_k = 1$ and $p_k > 0$.
*   $\boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_K]^\top$ are the concentration parameters, with $\alpha_k > 0$.

The probability density function is:
$$ f(\mathbf{p}; \boldsymbol{\alpha}) = \frac{1}{B(\boldsymbol{\alpha})} \prod_{k=1}^K p_k^{\alpha_k - 1} $$
where $B(\boldsymbol{\alpha})$ is the multivariate Beta function.

### 1.2 Latent Gaussian Process Linkage
To incorporate input-dependent uncertainty, we link the concentration parameters $\alpha_k$ to latent functions. We employ $K$ independent Latent Gaussian Processes (LGPs), denoted as $f_1(\mathbf{x}), \dots, f_K(\mathbf{x})$.

Since $\alpha_k$ must be strictly positive, we cannot use the GP output directly. We employ a **Log-Normal approximation** to link the unbounded GP output to the positive domain. Specifically, we treat the concentration parameter $\alpha_k$ as a random variable whose logarithm follows a Gaussian distribution defined by the GP:

$$ \ln \alpha_k(\mathbf{x}) \approx f_k(\mathbf{x}) $$
$$ f_k(\mathbf{x}) \sim \mathcal{GP}(\mu_k(\mathbf{x}), k_k(\mathbf{x}, \mathbf{x}')) $$

To make inference tractable, we utilize the moment-matching properties of the Log-Normal distribution. If $Z \sim \text{LogNormal}(\mu, \sigma^2)$, its mean and variance are:
$$ \mathbb{E}[Z] = e^{\mu + \sigma^2/2} $$
$$ \text{Var}[Z] = (e^{\sigma^2} - 1) e^{2\mu + \sigma^2} $$

In our implementation, we invert these relationships to map a desired "observed" alpha to the latent Gaussian space (mean and variance).

### 1.3 Observation Model (Likelihood)
We treat the observed one-hot labels $\mathbf{y} \in \{0, 1\}^K$ as evidence that updates the Dirichlet parameters. We define a base prior $\alpha_\epsilon$ (representing weak initial belief) and an observation strength $s$.

The "posterior" concentration parameters given an observation $\mathbf{y}$ are modeled as:
$$ \alpha_k = \alpha_\epsilon + s \cdot y_k $$

*   If class $k$ is observed ($y_k=1$), $\alpha_k$ increases by $s$, shifting the mass of the Dirichlet distribution towards corner $k$.
*   If class $k$ is not observed ($y_k=0$), $\alpha_k$ remains at the baseline $\alpha_\epsilon$.

### 1.4 Uncertainty Decomposition
A critical feature of GPP is the ability to decompose the total uncertainty of the prediction into distinct components. We utilize **Information Theoretic** measures for this purpose.

Let $\mathbf{p}$ be the random probability vector drawn from the posterior Dirichlet distribution. The predictive distribution for a new label $\hat{y}$ is the expected categorical distribution:
$$ P(\hat{y}=k) = \bar{p}_k = \mathbb{E}_{\mathbf{p}} [p_k] = \frac{\alpha_k}{\sum_j \alpha_j} $$

**1. Total Uncertainty (Entropy of the Mean):**
Represents the uncertainty in the final prediction, marginalized over all model parameters.
$$ H[\bar{\mathbf{p}}] = - \sum_{k=1}^K \bar{p}_k \ln \bar{p}_k $$

**2. Aleatoric Uncertainty (Expected Entropy):**
Represents the expected ambiguity of the data itself. Even if we knew the true distribution $\mathbf{p}$ perfectly, there would still be entropy due to class overlap/noise.
$$ \text{Alea} = \mathbb{E}_{\mathbf{p}} [H[\mathbf{p}]] = \mathbb{E}_{\mathbf{p}} \left[ - \sum_{k=1}^K p_k \ln p_k \right] $$

**3. Epistemic Uncertainty (Mutual Information):**
Represents the reduction in uncertainty about the label $\hat{y}$ given knowledge of the true parameters $\mathbf{p}$. This captures the "model uncertainty"—how much the distribution $\mathbf{p}$ spreads over the simplex.
$$ \text{Epist} = I(\hat{y}; \mathbf{p}) = H[\bar{\mathbf{p}}] - \mathbb{E}_{\mathbf{p}} [H[\mathbf{p}]] $$
$$ \text{Total Uncertainty} - \text{Aleatoric Uncertainty} $$

---

## 2. Implementation Details

The implementation is located in `GPax/probing/gp_multiclass.py`. We walk through the exact code corresponding to the mathematical steps above.

### 2.1 Latent Space Transformation
The function `get_latent_observations_dirichlet` implements the mapping from observed labels to the latent GP space (Eq. 1.2 and 1.3).

```python
def get_latent_observations_dirichlet(params, y, warp_func=None):
  # 1. Retrieve Hyperparameters
  alpha_eps, strength = retrieve_params(params, ['alpha_eps', 'strength'], warp_func)
  
  # 2. Compute Target Alphas (Eq. 1.3)
  # alpha_k = alpha_eps + s * y_k
  alpha = jnp.ones_like(y) * alpha_eps + y * strength
  
  # 3. Map to Latent Gaussian Moments (Inverting Log-Normal moments)
  # We define the latent variance (var) and mean (mu) such that
  # the LogNormal(mu, var) has mean = alpha and "compatible" variance.
  # Specifically, we use the transformation:
  # var = log(1/alpha + 1)
  # mu = log(alpha) - var/2
  var, y_latent = get_latent_var_mu_dirichlet(alpha)
  
  return y_latent, var
```

### 2.2 Prediction Loop
The prediction involves training $K$ independent Gaussian Processes. This is handled in `dirichlet_gp_predict`.

```python
def dirichlet_gp_predict(..., x_query, x_observed, y_observed, ...):
  # 1. Transform Observations to Latent Space
  y_latent, var_latent = get_latent_observations_dirichlet(params, y_observed, ...)
  
  predictions = []
  
  # 2. Iterate over each class k = 1...K
  for i in range(num_classes):
    # Slice the observations for class i
    y_observed_i = y_latent[:, i:i+1]
    var_observed_i = var_latent[:, i]
    
    # 3. Standard GP Prediction
    # We use the exact GP inference treating the latent 'var' as heteroscedastic noise
    mu_var = gp_predict(
        ..., 
        x_query=x_query, 
        x_observed=x_observed, 
        y_observed=y_observed_i, 
        var_observed=var_observed_i, 
        ...
    )
    predictions.append(mu_var)
    
  return predictions
```

### 2.3 Uncertainty Quantification (Monte Carlo)
Since the Dirichlet distribution and the Entropy terms do not have simple closed forms under the Log-Normal mixture, we use Monte Carlo sampling in `classifier_samples_uncertainty_multiclass`.

```python
def classifier_samples_uncertainty_multiclass(p_samples):
  """
  p_samples: Shape (N_query, K, N_samples)
  Contains sampled probability vectors p drawn from the posterior.
  """
  
  # 1. Judged Probability (Mean Prediction)
  mu = jnp.mean(p_samples, axis=2) 
  
  # 2. Aleatoric Uncertainty (Expected Entropy)
  # Calculate entropy for each sample vector p^(s)
  # H(p^(s)) = - sum(p_k^(s) * log(p_k^(s)))
  sample_entropies = -jnp.sum(p_samples * jnp.log(p_samples + eps), axis=1)
  # Average over samples
  categorical_entropy = jnp.mean(sample_entropies, axis=1, keepdims=True)
  
  # 3. Total Uncertainty (Entropy of Mean)
  # H(mu) = - sum(mu_k * log(mu_k))
  entropy_of_mean = -jnp.sum(mu * jnp.log(mu + eps), axis=1, keepdims=True)
  
  # 4. Epistemic Uncertainty (Mutual Information)
  # I = H(mu) - E[H(p)]
  info_gain = entropy_of_mean - categorical_entropy
  
  return {
      'expected_aleatory_entropy': categorical_entropy, # Aleatoric
      'information_gain': info_gain,                    # Epistemic
      'categorical_mu': mu,                             # Prediction
      ...
  }
```

---

## 3. Experimental Verification (Synthetic Shapes)

To rigorously verify the implementation, we utilized the **3DShapes dataset**. This dataset contains 480,000 synthetic images of simple 3D shapes (cube, sphere, cylinder, capsule) with varying factors of variation (floor hue, wall hue, object hue, scale, shape, orientation).

### 3.1 Experimental Setup
The verification logic is contained in `gpp_extended_verification.py`.

#### Models
We trained three distinct Convolutional Neural Networks (CNNs) to serve as embedding generators:
1.  **M1 (Concept Aligned)**: A standard CNN trained to predict *all* generative factors (shape, scale, color, etc.). It has a rich, disentangled representation. (Embedding dim: 64).
2.  **M2 (Shape Specialist)**: A small CNN trained *only* to predict the shape. It should have near-perfect performance on shape tasks but fail on color tasks. (Embedding dim: 8).
3.  **M3 (Color Specialist)**: A small CNN trained *only* to predict the floor hue. It should fail completely on shape tasks. (Embedding dim: 8).

#### Probing Task: "Multi_P2_Shape"
We defined a 3-class classification task:
*   **Target**: Predict the object shape.
*   **Classes**: Cube (0), Sphere (1), Cylinder (2). Capsules (3) were excluded to create a clean 3-way problem.
*   **Input**: The frozen embeddings from models M1, M2, or M3.
*   **Probe**: The Dirichlet-GPP model described above.

#### Procedure
For each model (M1, M2, M3) and sample size $N \in \{2, 4, 8, 16, 32, 64, 128\}$:
1.  Sample $N$ observations from the training set.
2.  Train the GPP-Dirichlet probe on these $N$ observations.
3.  Evaluate AUROC (Area Under ROC Curve) on a held-out test set.
4.  Compare against baselines: **LPE** (Linear Probe Ensemble), **SVM**, and **LP** (Standard Linear Probe).

### 3.2 Results

The results demonstrate the validity of the implementation. M2 (the Shape Specialist) achieves near-perfect performance rapidly. M1 (Generalist) learns steadily. M3 (Color Specialist) fails as expected, confirming the probe is measuring the representation content, not hallucinating.

#### Table 1: AUROC Results for Multi_P2_Shape (3-Class Classification)

| Sample Size (N) | Model | GPP-Dirichlet | LPE (Baseline) | LP (Baseline) |
| :--- | :--- | :--- | :--- | :--- |
| **N=2** | M1 (General) | 0.553 | 0.554 | 0.528 |
| | M2 (Shape) | 0.771 | 0.698 | 0.588 |
| | M3 (Color) | 0.502 | 0.503 | 0.501 |
| **N=4** | M1 (General) | 0.567 | 0.553 | 0.567 |
| | M2 (Shape) | 0.854 | 0.812 | 0.870 |
| | M3 (Color) | 0.501 | 0.504 | 0.500 |
| **N=8** | M1 (General) | 0.612 | 0.612 | 0.621 |
| | M2 (Shape) | 0.870 | 0.895 | 0.916 |
| | M3 (Color) | 0.497 | 0.503 | 0.499 |
| **N=16** | M1 (General) | 0.691 | 0.732 | 0.735 |
| | M2 (Shape) | 0.976 | 0.989 | 0.997 |
| | M3 (Color) | 0.509 | 0.510 | 0.513 |
| **N=32** | M1 (General) | 0.780 | 0.801 | 0.784 |
| | M2 (Shape) | 0.999 | 0.999 | 0.999 |
| | M3 (Color) | 0.498 | 0.531 | 0.537 |
| **N=64** | M1 (General) | 0.839 | 0.878 | 0.849 |
| | M2 (Shape) | 1.000 | 1.000 | 1.000 |
| | M3 (Color) | 0.511 | 0.594 | 0.595 |
| **N=128** | M1 (General) | 0.864 | 0.895 | 0.888 |
| | M2 (Shape) | 1.000 | 1.000 | 1.000 |
| | M3 (Color) | 0.528 | 0.631 | 0.631 |

*Values represent the average AUROC across 5 random seeds per sample size.*

### 3.3 Uncertainty Manifold Verification
In addition to accuracy, we verified the topological structure of the uncertainty. We projected the predicted probability vectors onto the 2-simplex (triangle) for the 3-class problem.

*   **Epistemic Uncertainty**: High in regions far from training data points (corners of the simplex or empty regions).
*   **Aleatoric Uncertainty**: High in the center of the simplex (where $p_1 \approx p_2 \approx p_3 \approx 1/3$), representing inherent ambiguity between shapes.

The simulation scripts (`simulate_ambiguous_data` in `gpp_extended_verification.py`) generated "morphed" embeddings between class centroids to artificially induce aleatoric uncertainty. The GPP-Dirichlet probe correctly identified this as high Aleatoric and low Epistemic uncertainty, whereas distance-based metrics often conflated the two.

---

## Appendix: Comprehensive GPP-Dirichlet Implementation Details

This section provides a mathematically rigorous, differential analysis of every step taken to transform the original binary GPP (`GPax/probing/gp.py`) into the Dirichlet-based multiclass extension (`GPax/probing/gp_multiclass.py`).

### A.1 Transformation of the Latent Observation Model

In GPP, discrete class labels are not modeled directly. Instead, they are transformed into "latent observations" for the underlying Gaussian Processes. This update generalizes the Beta-based transform to a Dirichlet-based one.

#### Original: Beta Transform (Binary)
In `GPax/probing/gp.py` (`get_latent_observations`):
The binary labels $y \in \{0, 1\}$ update a 2-parameter Beta distribution $\text{Beta}(\alpha, \beta)$.
$$ [\alpha, \beta] = [\alpha_\epsilon, \alpha_\epsilon] + [y, 1-y] \cdot s $$
The GP targets ($\mu, \sigma^2$) are derived via log-normal matching:
$$ \sigma^2_{latent} = \ln(1/\alpha + 1) $$
$$ \mu_{latent} = \ln(\alpha) - \sigma^2_{latent}/2 $$

#### New: Dirichlet Transform (Multiclass)
In `GPax/probing/gp_multiclass.py` (`get_latent_observations_dirichlet`):
The labels are now one-hot vectors $\mathbf{y} \in \{0, 1\}^K$. The update applies to the concentration vector $\boldsymbol{\alpha} = [\alpha_1, \dots, \alpha_K]$.
$$ \alpha_k = \alpha_\epsilon + y_k \cdot s $$

**Code Change:**
```python:GPax/probing/gp_multiclass.py
def get_latent_observations_dirichlet(params, y, warp_func=None):
    # ...
    # Original (implicitly 2D):
    # alpha = jnp.ones((y.shape[0], 2)) * alpha_eps + jnp.hstack([y, 1 - y]) * strength
    
    # New (explicitly K-dimensional):
    alpha = jnp.ones_like(y) * alpha_eps + y * strength
    
    # The Log-Normal moment matching function `get_latent_var_mu_dirichlet` 
    # uses the same formula but broadcasts over K dimensions instead of 2.
    var, y = get_latent_var_mu_dirichlet(alpha) 
    return y, var
```

### A.2 Generalization of the Prediction Loop

The core computational engine of GPP is a set of independent Gaussian Processes.

#### Original: Fixed Binary Loop
In `beta_gp_predict`, the code hardcodes a loop over range(2).
```python:GPax/probing/gp.py
def beta_gp_predict(...):
  # ...
  for i in range(2):
    # predict for class 0 and class 1
```

#### New: Dynamic Multiclass Loop
In `dirichlet_gp_predict`, the loop dynamically adjusts to the number of classes $K$ present in the label matrix.
```python:GPax/probing/gp_multiclass.py
def dirichlet_gp_predict(..., y_observed, ...):
    # ...
    num_classes = y_latent.shape[1]
    for i in range(num_classes):
        # Slice observations for class k
        y_observed_i = y_latent[:, i:i+1]
        # ...
        # Call the exact same base 'gp_predict' function
        mu_var = gp_predict(..., y_observed=y_observed_i, ...)
        predictions.append(mu_var)
```
**Mathematical Implication:** This confirms that the $K$ latent functions $f_1, \dots, f_K$ are modeled as *a priori* independent GPs. The correlation between classes arises solely from the softmax/normalization step in the posterior sampling, not from the kernel itself (which remains a shared cosine kernel).

### A.3 Uncertainty Quantification: The Shift to Information Theory

The most significant mathematical update is in how uncertainty is measured. The binary case used Bernoulli variance/entropy; the multiclass case requires Mutual Information on the Simplex.

#### Original: Bernoulli Metrics
In `classifier_samples_uncertainty`:
*   `bernoulli_entropy`: $- [p \ln p + (1-p) \ln (1-p)]$
*   `info_gain`: $H(\bar{p}) - \mathbb{E}[H(p)]$

#### New: Categorical Metrics
In `classifier_samples_uncertainty_multiclass`:
The samples `p_samples` have shape $(N_{query}, K, N_{samples})$.

1.  **Aleatoric Uncertainty (Expected Entropy):**
    $$ \mathbb{E}[H(\mathbf{p})] = \frac{1}{S} \sum_{s=1}^S \left( - \sum_{k=1}^K p_k^{(s)} \ln p_k^{(s)} \right) $$
    
    ```python:GPax/probing/gp_multiclass.py
    # Compute entropy for EACH sample vector first
    sample_entropies = -jnp.sum(p_samples * jnp.log(p_samples + eps), axis=1)
    # Then average
    categorical_entropy = jnp.mean(sample_entropies, axis=1, keepdims=True)
    ```

2.  **Epistemic Uncertainty (Mutual Information):**
    First, compute the entropy of the mean prediction $\bar{\mathbf{p}}$:
    $$ H(\bar{\mathbf{p}}) = - \sum_{k=1}^K \bar{p}_k \ln \bar{p}_k $$
    
    ```python:GPax/probing/gp_multiclass.py
    entropy_of_mean = -jnp.sum(mu * jnp.log(mu + eps), axis=1, keepdims=True)
    ```
    
    Then, the Mutual Information (Epistemic Uncertainty) is the difference:
    ```python
    info_gain = entropy_of_mean - categorical_entropy
    ```

### A.4 API Surface Update

To expose this functionality, a new high-level wrapper `gpp_multiclass` was created in `GPax/probing/probabilistic_probe_multiclass.py`. It mirrors the signature of `gpp` but accepts an optional `num_classes` argument and handles the expanded one-hot label format.

```python:GPax/probing/probabilistic_probe_multiclass.py
def gpp_multiclass(x_query, x_observed, y_observed, num_classes=None, ...):
    # 1. Setup params including num_classes
    params = {'alpha_eps': alpha_eps, 'strength': strength}
    if num_classes: params['num_classes'] = num_classes
    
    # 2. Call the new predictor
    predictions = gp.dirichlet_gp_predict(..., params=params)
    
    # 3. Call the new uncertainty quantifier
    measures = gp.dirichlet_gp_uncertainty(predictions, ...)
    return measures
```
