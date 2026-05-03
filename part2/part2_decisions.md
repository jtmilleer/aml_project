# Part II — Design Decisions

## Dataset Characteristics (what drove every decision)

| Property | Value |
|---|---|
| Samples | 158 total (126 class 0, 32 class 1) |
| Class imbalance | ~4:1 ratio — 20% positive |
| Features | 125 continuous, no missing values |
| Highly correlated feature pairs (|r| > 0.90) | **484 pairs** (e.g. X9↔X59 at r=0.9989) |
| PCA components for 95% variance | **11** (from 125) |
| Feature scales | Widely varying (std range: 0.04 – 9.36) |

The massive redundancy (125 raw features → 11 effective dimensions) and tiny sample size are the dominant constraints shaping both methods.

---

## Traditional Method — SVM + PCA (`part2_traditional.py`)

### Pipeline: `SMOTE → StandardScaler → PCA(95%) → SVC(RBF)`

### Why SVM?

SVMs work well in **small-sample, high-dimensional settings** because they find the maximum-margin hyperplane — a geometrically principled decision boundary that generalizes from limited data. The number of support vectors (not the total sample count) governs the solution, so SVMs can perform well even with 158 samples.

The **RBF (Gaussian) kernel** maps data into an infinite-dimensional space, allowing nonlinear decision boundaries without explicitly engineering features. This is appropriate because we have no prior knowledge of whether classes are linearly separable.

`class_weight='balanced'` multiplies the penalty C by `n_samples / (n_classes × class_count)` for each class, giving the minority class more influence on the margin — a clean built-in way to handle imbalance.

### Why PCA?

With 484 feature pairs correlated above 0.90, the data is nearly degenerate. PCA **decorrelates** the features and reveals that 95% of the variance lives in just 11 dimensions. This matters for three reasons:

1. **Multicollinearity** — redundant features make the SVM optimization unstable and inflate the effective feature space.
2. **Speed** — SVM training complexity scales with the number of features; going from 125 to ~11 dimensions is a large speedup.
3. **Generalization** — noise dimensions (the remaining 5% variance) could hurt test performance; PCA discards them.

`n_components=0.95` is chosen (rather than a fixed integer) so the threshold adapts if the dataset changes.

### Why SMOTE?

With 32 positive samples, a naive classifier that always predicts class 0 achieves 80% accuracy. SMOTE (Synthetic Minority Over-sampling Technique) synthesizes new minority-class samples by **interpolating between existing ones in feature space**, rather than simply duplicating them. This gives the model a richer view of the minority class boundary.

`k_neighbors=4` (down from the default 5) is used because each training fold in 5-fold CV only has ~25 minority samples — a neighborhood of 5 would occasionally reach the full set, so 4 is safer.

**Crucially**, SMOTE is placed inside the `imblearn.Pipeline` so it is applied **only to training folds during cross-validation** and never to validation folds. This prevents data leakage.

### GridSearchCV — Hyperparameter Choices

| Parameter | Values searched | Rationale |
|---|---|---|
| `C` | 0.01, 0.1, 1, 10, 100 | Log-spaced; small C = soft margin (regularization), large C = hard margin (low bias). With a tiny dataset we expect the optimal C to be moderate. |
| `gamma` | 'scale', 'auto', 0.01, 0.1 | Controls the RBF bandwidth. `'scale'` (= 1/(n_features × var)) is usually a good starting point; we also search explicit values to cover broader/narrower kernels. |
| CV strategy | 5-fold StratifiedKFold | Stratified folds preserve the 4:1 class ratio in every fold. 5 folds is standard; fewer folds → high variance estimate, more folds → computationally expensive with 20 hyperparameter combos. |
| Scoring metric | `f1_macro` | Accuracy is misleading at 4:1 imbalance. F1-macro treats both classes equally and penalizes poor recall on the minority class. |

---

## Deep Learning Method — MLP (`part2_deep.py`)

### Why an MLP?

While the dataset is small (which favors traditional methods), an MLP is included as a comparison because:
- It can learn **nonlinear interactions** between features that a kernel SVM approximates implicitly.
- With heavy regularization it can still generalize from 158 samples.
- Deep learning methods are required for Part I and are worth benchmarking here.

### Architecture: `125 → 128 → 64 → 32 → 1`

| Component | Choice | Reason |
|---|---|---|
| 3 hidden layers | 128 → 64 → 32 | Tapering width (each layer halves the previous) is a standard heuristic that compresses representations progressively. A single wide layer would lack depth; too many layers would overfit with 158 samples. |
| BatchNorm | After each linear layer | Normalizes activations to zero mean/unit variance per batch, stabilizing training and acting as a mild regularizer. Also reduces sensitivity to weight initialization. |
| ReLU | Activation | Standard choice; avoids vanishing gradients and is computationally cheap. |
| Dropout(0.5) | After each BatchNorm+ReLU | Randomly zeros 50% of neurons during training. At 0.5, this is aggressive regularization — appropriate because the dataset is very small and overfitting is the main risk. |
| Output layer | Single logit (no activation) | Used with `BCEWithLogitsLoss` which applies sigmoid internally for numerical stability. |

### Handling Class Imbalance (two-pronged)

1. **SMOTE** — same rationale as the traditional method. Applied per training fold to avoid leakage.
2. **`pos_weight = n_neg / n_pos ≈ 3.94`** — passed to `BCEWithLogitsLoss`. This scales the loss for positive samples, making each positive example count ~4× as much as a negative one. Using both SMOTE and pos_weight provides complementary regularization: SMOTE fixes the input distribution, pos_weight fixes the gradient scale.

### Optimizer and Training Schedule

| Setting | Value | Reason |
|---|---|---|
| Optimizer | Adam, lr=1e-3 | Adam adapts per-parameter learning rates; well-suited for noisy gradients from a small dataset. lr=1e-3 is the standard default. |
| Weight decay | 1e-4 | L2 regularization on all weights. Penalizes large weights and reduces overfitting. |
| LR scheduler | StepLR (step=100, γ=0.5) | Halves the LR every 100 epochs. Allows the model to take larger steps early (exploration) and fine-tune later (convergence). |
| Epochs | 300 | With the step scheduler, the LR drops to 1e-3 → 5e-4 → 2.5e-4 → 1.25e-4 over 300 epochs. Empirically sufficient for convergence on this data size. |
| Batch size | 32 | After SMOTE, training data is ~400+ samples; batch size 32 gives ~12 gradient updates/epoch. Small enough for useful stochasticity (implicit regularization), large enough for BatchNorm to estimate stable statistics. |

### Cross-Validation

5-fold StratifiedKFold is used (same as the traditional method) for a fair comparison of CV F1 and AUC-ROC. A separate model is trained per fold (no shared weights). The final model is then retrained on the full development set.

---

## Why Traditional (SVM+PCA) is Expected to Win

| Factor | Favors SVM+PCA | Favors MLP |
|---|---|---|
| Sample size (158) | ✓ SVMs need fewer samples | — small datasets → MLP overfits |
| Effective dimensionality (11 PCA dims) | ✓ SVMs excel in low-dim spaces | — |
| Interpretability | ✓ Clear support vector picture | — black box |
| Complex nonlinear interactions | — | ✓ MLP can model these |
| Large data / complex patterns | — | ✓ MLP scales better |

With only 158 samples and 11 effective dimensions, **SVM+PCA is the primary submission method**. The MLP is a meaningful alternative if the SVM underperforms on the instructor's held-out test set, but overfitting risk is high.

---

## Iteration 2 — What the Experiments Revealed

After running both v1 models and a systematic improvement sweep, several important things were learned.

### What the exploration script tested

- **5 classifiers** (SVM+PCA at two param settings, LogReg+PCA, RandomForest, HistGBM) with both SMOTE and SMOTEENN
- **PCA component thresholds** (0.90, 0.95, 0.99, and fixed counts 5–20) on SVM
- **3 resampling strategies** (SMOTE, SMOTEENN, ADASYN)
- **5 MLP configurations** with and without PCA preprocessing

### Key finding 1 — SMOTEENN hurts SVM with low C/gamma

| Resampler | SVM params | CV F1 |
|---|---|---|
| SMOTE | C=0.01, γ=0.01 | **0.660** |
| SMOTEENN | C=0.01, γ=0.01 | 0.334 (catastrophic) |
| SMOTEENN | C=1, γ=scale | 0.661 (recovered) |
| ADASYN | any | 0.168 (unusable) |

**Why**: SMOTEENN's ENN cleaning step removes borderline minority-class samples — the very samples closest to the decision boundary that are most informative for a small-margin SVM. With soft-margin C=0.01 the model is especially sensitive to this loss. Critically, the PCA component count made **zero difference** to SVM F1 when SMOTEENN was used — all component counts gave the same 0.334. This confirmed the resampler, not the dimensionality, was the bottleneck. **Decision: keep SMOTE, discard SMOTEENN.**

### Key finding 2 — Expanded hyperparameter grid for SVM

The v1 grid found `C=0.01, gamma=0.01`. The improved v2 grid (8 C values × 7 gamma values = 56 combos) found `C=0.1, gamma=0.001`.

| Version | Best params | CV F1 |
|---|---|---|
| v1 (20 combos) | C=0.01, γ=0.01 | 0.660 |
| v2 (56 combos) | C=0.1, γ=0.001 | **0.676** |

The wider search yielded a +0.016 F1 improvement. The AUC artifact (v1 reported 0.256 on training) was also resolved — v2 shows AUC=0.763, because the SMOTEENN→SMOTE switch removes the bad interaction with Platt scaling.

### Key finding 3 — Threshold optimisation (out-of-fold)

Rather than using 0.5 as the classification threshold, the improved code collects out-of-fold probabilities across all 5 CV folds and sweeps 181 thresholds (0.05–0.95) to find the one maximising F1 macro. For the traditional model the optimal threshold was **0.515** — barely different from 0.5 — so the gain was minimal here. For the deep model the optimal threshold was **0.945**, which transformed:

| | Default threshold (0.5) | Optimal threshold (0.945) |
|---|---|---|
| OOF F1 macro | 0.334 | **0.508** |

This large shift revealed the root cause of the deep model's low AUC: see finding 4.

### Key finding 4 — The `pos_weight` bug in the deep model

In v1, `pos_weight` was computed **after SMOTE** resampled the training fold:
```python
n_neg, n_pos = np.bincount(y_tr_smote)   # ≈ equal → pos_weight ≈ 1.0
```
After SMOTE, classes are balanced, so `pos_weight ≈ 1.0` — effectively disabled. The model then learned to output **low probabilities for the positive class** (the minority), because nothing in the loss function penalised it. This produced inverted AUC scores (< 0.5 means the model ranked negatives above positives).

The fix in v2: compute `pos_weight` from the **original** training fold class ratio (~4.0) and apply it on top of SMOTE. Now the loss genuinely penalises false negatives even after SMOTE balances the counts.

### Key finding 5 — MLP still overfits despite all fixes

Even with PCA (125 → 10 components), smaller network (64→32), corrected pos_weight, CosineAnnealingLR, and threshold optimisation, the MLP training F1 (0.94) far exceeds OOF F1 (0.51). The gap is ~0.43 — classic overfitting. **158 samples is simply too few for gradient-based learning** to generalise reliably without far more aggressive regularisation or additional data.

### Summary of improvements

| | v1 CV F1 | v2 CV F1 | Change |
|---|---|---|---|
| Traditional (SVM+PCA) | 0.660 | **0.676** | +0.016 |
| Traditional AUC | 0.256 (broken) | **0.763** (correct) | fixed |
| Deep (MLP) OOF F1 | 0.318 | **0.508** | +0.190 |
| Deep AUC | ~0.31 (inverted) | ~0.36 (still low) | partial fix |

The deep model's OOF F1 gains are largely from threshold optimisation, not genuine generalisation. **SVM+PCA remains the submission method.**
