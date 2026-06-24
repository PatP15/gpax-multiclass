# AnnoMI Pipeline: Multiclass GPP with Context & Cascading Uncertainty

This directory contains the pipeline for analyzing the **AnnoMI** dataset using **Dirichlet Gaussian Process Probing (GPP)**. The goal is to classify client motivation (Change, Sustain, Neutral) and evaluate how **Therapist Context** and **Quality Estimation** affect model uncertainty.

## Experiments

We run 4 experimental setups, varying the training sample size ($N \in [10, 30, 50, 75, 100]$):

1.  **Client Only (Baseline)**:
    *   **Input**: Client's utterance text.
    *   **Model**: Gemma-2-2b Embeddings -> PCA (64D) -> GPP-Dirichlet (3-class).
    *   **Goal**: Establish baseline performance.

2.  **Context (Therapist + Client)**:
    *   **Input**: `Therapist: <text> \n Client: <text>`
    *   **Model**: Gemma-2-2b Embeddings -> PCA -> GPP-Dirichlet.
    *   **Goal**: Test if adding therapist context reduces Epistemic Uncertainty.

3.  **Therapist Quality Classifier**:
    *   **Input**: Therapist's utterance text.
    *   **Task**: Classify "High Quality" (Reflection, Open Question) vs. "Low Quality" (Advising, Closed).
    *   **Model**: GPP-Beta (Binary).
    *   **Output**: Predicted Quality Label + Uncertainty (Aleatoric, Epistemic).

4.  **Cascading Uncertainty**:
    *   **Input**: Augmented text `[Quality: <High/Low>] Therapist: ... Client: ...`
    *   **Features**: Concatenation of [Text Embeddings (64D), Quality Uncertainty (2D)].
    *   **Model**: GPP-Dirichlet (3-class).
    *   **Goal**: Assess if explicitly feeding the *uncertainty* of the upstream quality model improves the downstream motivation classification or calibration.

## Mathematical Details

### Dirichlet GPP (Multiclass)
We extend the Binary GPP (Beta-Bernoulli) to Multiclass (Dirichlet-Categorical).
- **Prior**: $\alpha \sim \text{Dirichlet}(\alpha_\epsilon)$
- **Likelihood**: $y \sim \text{Categorical}(\pi)$
- **Uncertainty**:
    - **Aleatoric**: $\mathbb{E}[H(y|\pi)]$ (Expected Entropy of data noise)
    - **Epistemic**: $-\mathbb{H}[\pi]$ (Entropy of the distribution over distributions, approximated)

### Cascading Uncertainty
In Experiment 4, we define the feature vector $x_{\text{final}}$ as:
$$ x_{\text{final}} = [ \phi(\text{AugmentedText}), \mathbb{H}_{\text{aleatoric}}(q), \mathbb{H}_{\text{epistemic}}(q) ] $$
where $q$ is the upstream quality prediction. This allows the Motivation GPP to learn correlations like: *"When the quality model is uncertain, rely less on the therapist's text."*

## Running the Pipeline

1.  **Install Requirements**:
    The script `run_annomi.sh` will install dependencies from `requirements.txt`.
    ```bash
    ./annoMI/run_annomi.sh
    ```

2.  **Output**:
    -   **Figures**: `annoMI/figures/`
        -   `accuracy_comparison.png`: Accuracy vs Sample Size.
        -   `uncertainty_comparison.png`: Epistemic/Aleatoric Uncertainty trends.
    -   **Logs**: Console output includes qualitative examples.

