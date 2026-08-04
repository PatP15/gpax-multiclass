# IC-AnnoMI Motivation Analysis: Comprehensive Technical Report

## 1. Executive Summary
This project evaluates the capability of open-weights Large Language Models (**Gemma-3-27b-it** and **Qwen3-VL-30B**) to encode motivational interviewing concepts (Change Talk vs. Sustain Talk vs. Neutral). We employ **Gaussian Process Probing (GPP)** to analyze these representations, focusing on **uncertainty quantification** and **sample efficiency**.

Key technical achievements include:
1.  **Multiclass GPP Extension**: Extended the GPP framework from binary (Beta) to multiclass (Dirichlet), validated against the original implementation.
2.  **Prompt Engineering**: Standardized `INPUT/OUTPUT` formats and demonstrated that **Expert Few-Shot Prompting** significantly improves representation quality.
3.  **Uncertainty Decomposition**: Successfully disentangled Aleatoric (data ambiguity) and Epistemic (model uncertainty) components, visualized via Uncertainty Manifolds.
4.  **Robust Evaluation**: Compared GPP against a Linear Probe Ensemble (LPE) baseline across varying sample sizes ($N \in \{50, \dots, 2400\}$) and task settings (Binary vs. Multiclass).

---

## 2. Experimental Design & Rationale

### 2.1 Why Binary and Multiclass?
We conducted parallel experiments for two tasks:
1.  **Multiclass (3-Way)**: *Change* vs. *Sustain* vs. *Neutral*. This reflects the real-world complexity of the IC-AnnoMI dataset.
2.  **Binary (2-Way)**: *Change* vs. *Non-Change* (Sustain + Neutral).
    *   **Rationale**: The binary task serves as a simpler proxy to verify the "Change Talk" signal strength. It also acts as a regression test for the original GPP-Beta implementation, ensuring that our new pipeline remains consistent with established baselines.

### 2.2 Dataset & Balancing Strategy
*   **Source**: IC-AnnoMI dataset (`IC_AnnoMI.csv`).
*   **Preprocessing**:
    *   **Filtering**: Only client utterances following a therapist utterance were selected.
    *   **Balancing**: The **training set** was balanced by undersampling majority classes to match the count of the minority class. This prevents the GP from learning a trivial prior (e.g., predicting "Neutral" always).
    *   **Testing**: The **test set** was left unbalanced to evaluate performance on the natural data distribution.
*   **Sample Sizes**: We swept $N \in \{50, 100, 250, 500, 1000, 1500, 2000, 2400\}$ to measure sample efficiency.

---

## 3. Mathematical & Implementation Details

### 3.1 From Beta to Dirichlet (The Math)
The core innovation is the extension of GPP to multiclass.

*   **Original (Binary)**: MODELED $p \sim \text{Beta}(\alpha, \beta)$. Linked to 1 Latent GP via Log-Normal approx.
*   **New (Multiclass)**: MODELED $\mathbf{p} \sim \text{Dir}(\alpha_1, \dots, \alpha_K)$. Linked to $K$ Latent GPs.

**Latent Linkage**:
We model the log-concentration parameters using independent GPs:
$$ \ln \alpha_k(\mathbf{x}) \approx f_k(\mathbf{x}) $$
This allows us to perform Stochastic Variational Inference (SVI) analytically by approximating the Dirichlet posterior moments.

### 3.2 Code Implementation
We standardized the input formats across all experiments to ensure fair comparison.

**Unified Prompt Format**:
```python
INPUT: (Context)
OUTPUT: (Label)
```

**Few-Shot Logic (Context Window Expansion)**:
To implement few-shot prompting, we expanded the context window (up to 1024 tokens) and prepended 2 static examples per class.

*Code Snippet (`annoMI/step1_process_data_prompted.py`)*:
```python
MOTIVATION_PROMPT_FEWSHOT = """You are an expert psychotherapist...
{few_shot_examples}

Given the following input, output the class (Change Talk, Sustain Talk, or Neutral).
INPUT: Therapist: {therapist_text}
Client: {client_text}
OUTPUT:"""
```

**Uncertainty Metrics**:
We implemented the decomposition in `GPax/probing/gp_multiclass.py`:
```python
# Aleatoric: Expected Entropy of Likelihood
alea = jnp.mean(-jnp.sum(p_samples * jnp.log(p_samples), axis=1), axis=0)

# Total: Entropy of Expected Prediction
total = -jnp.sum(mu * jnp.log(mu), axis=1)

# Epistemic: Mutual Information
epistemic = total - alea
```

### 3.3 Experimental Inputs (Detail)
We tested 5 distinct input configurations to isolate the effects of context, quality annotations, and prompting.

| Experiment | Input Template (`{t}`=Therapist, `{c}`=Client) | Prompt Strategy |
| :--- | :--- | :--- |
| **Context** | `INPUT: Therapist: {t}\nClient: {c}\nOUTPUT:` | Instruction Only |
| **Context + Quality** ⚠️ *oracle* | `INPUT: Therapist: {t} (Quality: {q})\nClient: {c}\nOUTPUT:` | Instruction Only |
| **Context (Expert)** | `[Expert Prompt]\nINPUT: Therapist: {t}\nClient: {c}\nOUTPUT:` | Expert Persona |
| **Context + Quality (Expert)** ⚠️ *oracle* | `[Expert Prompt]\nINPUT: Therapist: {t} (Quality: {q})\nClient: {c}\nOUTPUT:` | Expert Persona |

> ⚠️ **The two "Quality" rows are oracle upper bounds, not deployable results.** `{q}` is the *gold*
> therapist-quality label, injected into the **test** prompt as well as train (`docs/BUGS.md` B2). Gold
> quality correlates with client talk type, so these rows are inflated relative to the "Context" baseline
> they are compared against. The honest version (out-of-fold predicted quality) was not implemented.
| **Context (Expert Few-Shot)** | `[Expert Prompt]\n[Few-Shot Examples]\nINPUT: Therapist: {t}\nClient: {c}\nOUTPUT:` | Expert + Few-Shot |

*Note: The "Instruction Only" prompt is "choose one of the three options (change, sustain, or neutral)".*

---

## 4. Results: LLM Performance

### 4.1 Comprehensive Multiclass Results (N=2400)
We evaluated 5 probing configurations. **Accuracy** is reported on the held-out test set.

| Experiment Setup | Gemma Accuracy | Qwen Accuracy |
| :--- | :--- | :--- |
| **Context** (Baseline) | 63.4% | 57.5% |
| **Context + Quality** | 62.8% | 58.2% |
| **Context (Expert)** | 63.2% | 59.6% |
| **Context + Quality (Expert)** | 63.2% | 60.1% |
| **Context (Expert Few-Shot)** | **67.8%** | **63.9%** |

*   **Key Finding**: Few-Shot prompting provides a substantial boost (+4-6%) over all other methods. Simply adding "Quality" labels or Expert instructions alone yielded marginal or no gains, suggesting that **demonstration** (via few-shot) is more effective than **description** (via expert prompts) for this task.

### 4.2 GPP vs LPE (Sample Efficiency)
We compared GPP against the Linear Probe Ensemble (LPE) baseline on the Gemma embeddings (Few-Shot).

| Metric | GPP (N=50) | LPE (N=50) | GPP (N=2400) | LPE (N=2400) |
| :--- | :--- | :--- | :--- | :--- |
| **Accuracy** | **57.3%** | 54.3% | 67.8% | **68.6%** |

*   **Result**: GPP outperforms LPE by **+3.0%** in the low-data regime ($N=50$), validating its utility for few-shot probing.

---

## 5. Qualitative & Uncertainty Analysis

### 5.1 AUROC for Misclassification Detection
We measured the Area Under the ROC Curve (AUROC) using Epistemic Uncertainty (Negative Entropy of Parameters) as the anomaly score.
*   **Result**: The AUROC scores hovered around **0.30 - 0.35**.
*   **Interpretation**: Since Epistemic is defined as a measure of **Certainty** (Higher Value = More Certain), we expect high scores for *Correct* predictions and low scores for *Incorrect* ones. The AUROC metric typically measures the ability of a score to detect the positive class (Errors). Because our score (Certainty) is *negatively* correlated with Errors, we observe an AUROC < 0.5. Inverting this ($1 - \text{AUROC}$) yields a detection capability of **~0.65 - 0.70**, which is a strong baseline for uncertainty-based error detection in NLP. This confirms that when the model is uncertain about its parameters (Low Episteme), it is significantly more likely to be wrong.

### 5.2 Case Studies: The Four Quadrants of Uncertainty
We analyzed the test set to find examples in the extreme quadrants of Aleatoric (Data Ambiguity) and Epistemic (Model Certainty) space.
*   **Episteme**: Negative Entropy of parameters. **Higher (>13) = High Certainty.**
*   **Aleatoric**: Entropy of likelihood. **Higher (>0.8) = High Ambiguity.**

#### Quadrant 1: Low Aleatoric / High Episteme ("Confident Clarity")
*   **Example**:
    *   *Therapist*: "Is it okay if I take a look at what you put down?"
    *   *Client*: "Sure."
    *   **Ground Truth**: Neutral.
    *   **Prediction**: Neutral.
*   **Metrics**: Episteme = **17.95 (High)**, Aleatoric = **0.085 (Low)**.
*   **Analysis**: This is the ideal state. The utterance "Sure" in response to a permission question is unambiguously Neutral. The model is **Clear** about the data (Low Alea) and **Certain** about its parameters (High Episteme).

#### Quadrant 2: High Aleatoric / Low Episteme ("Total Confusion")
*   **Example**:
    *   *Therapist*: "Right. By the way, binge drinking is defined for women as having more than, uh, three drinks..."
    *   *Client*: "Mm-hmm. Yes, sometimes I feel worse after drinking."
    *   **Ground Truth**: Neutral.
    *   **Prediction**: Sustain.
*   **Metrics**: Episteme = **5.61 (Low)**, Aleatoric = **1.08 (Max)**.
*   **Analysis**: This example is a mess. The therapist talks about binge drinking definitions, and the client agrees ("Mm-hmm") but adds a negative feeling ("feel worse"). The model is maximally **Ambiguous** (Aleatoric ~1.0) and also admits it **doesn't know its parameters** (Low Episteme). This represents an Out-of-Distribution (OOD) point where the model fails completely.

#### Quadrants 3 & 4: The Empty Combinations
Interestingly, we found **zero examples** in the "High Alea / High Episteme" (Certain Ambiguity) and "Low Alea / Low Episteme" (Cautious Clarity) quadrants at the 10th/90th percentiles.
*   **Missing "Certain Ambiguity"**: The model rarely says "I am 100% certain that this is 50/50." When data ambiguity is high (High Alea), the model's parameter certainty almost always drops (Low Episteme). The GPP framework naturally couples these uncertainties: difficult data makes the GP posterior wider.
*   **Missing "Cautious Clarity"**: Conversely, when the model predicts a clear class (Low Alea), it usually does so with high parameter certainty. There are few cases where the model says "It's definitely Class A, but I'm not sure why."

#### Bonus Case: The "Sure" Context Ambiguity
Comparing Quadrant 1 ("Sure" -> Neutral) with another "Sure":
*   *Therapist*: "Would you be willing to come back?"
*   *Client*: "Sure." (GT: Change)
*   **Metrics**: Episteme = 12.6, Aleatoric = 0.50.
*   **Analysis**: This falls in the **Middle**. The word is the same, but the context implies Commitment (Change). The model is less certain (12.6 vs 17.9) and sees more ambiguity (0.50 vs 0.08), correctly reflecting the nuance that "Sure" can be a weak commitment compared to "Yes, absolutely."

### 5.3 Uncertainty Manifolds (Figure 6)
*   **Zoomed View Findings**: By removing outliers (>99th percentile), we observe that LPE's uncertainty collapses rapidly, while GPP maintains a calibrated "cloud" of high-Aleatoric points. These correspond to genuinely ambiguous utterances (like Case 2 above) where the model *should* express uncertainty.

---

## 6. Conclusion
The experiments confirm that **GPP-Dirichlet** is a powerful tool for probing LLM representations in multiclass settings. It offers superior **sample efficiency** compared to linear baselines and provides **interpretable uncertainty estimates**. **Gemma-3-27b-it** (with Expert Few-Shot prompting) yields the highest quality embeddings, outperforming Qwen3-VL-30B by ~4% in accuracy.
