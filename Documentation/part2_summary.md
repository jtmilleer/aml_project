# Part II — Method Summary (v4)

## Dataset

| Property | Value |
|---|---|
| File | `PartII_dev.csv` |
| Samples | 158 |
| Features | X1–X125 (125 continuous, no missing values) |
| Class distribution | 126 class 0 / 32 class 1 (4:1 imbalance, 20.3% positive) |
| Correlated feature pairs (|r| > 0.90) | 484 — severe multicollinearity |
| Features needed for 95% variance (PCA) | 11 |

---

## Traditional Method

### Pipeline

```
StandardScaler → SelectKBest(f_classif, k=10) → LogisticRegression
```

1. **StandardScaler** — zero-mean, unit-variance normalisation. Required before `f_classif` (assumes normality) and before logistic regression (gradient scale).
2. **SelectKBest(f_classif, k=10)** — ANOVA F-test on each feature against Y. Retains the 10 most statistically discriminative features, discarding the ~115 near-zero predictors. Directly measures class-separating power, unlike PCA which maximises variance regardless of predictive value.
3. **LogisticRegression(C=0.1, class_weight='balanced')** — L2-penalised logistic regression. `class_weight='balanced'` weights the minority class by n_neg/n_pos ≈ 3.9×, handling imbalance without SMOTE. C=0.1 provides moderate regularisation appropriate for the small sample size.

**No SMOTE used** — class_weight handles imbalance cleanly without introducing synthetic samples.

### Model Selection

Four pipelines were compared using **RepeatedStratifiedKFold (5 splits × 5 repeats = 25 folds)**. Per-fold threshold optimisation maximises F1-macro on each validation fold.

| Model | CV F1 (macro) | CV AUC-ROC | Threshold |
|---|---|---|---|
| **LogReg + SelectKBest(k=10)** | **0.724 ± 0.066** | **0.738** | 0.626 |
| LogReg + SelectKBest(k=7) | 0.724 ± 0.071 | 0.738 | 0.617 |
| SVM(RBF) + SelectKBest(k=8) | 0.702 ± 0.070 | 0.701 | 0.298 |
| LogReg + PCA(95%) | 0.648 ± 0.064 | 0.618 | 0.612 |

The SelectKBest→LogReg pipeline outperforms PCA→LogReg by +0.076 F1. Selecting raw discriminative features is more effective than rotating the full feature space.

### Results

| Metric | CV estimate | Training (full dev set) |
|---|---|---|
| Accuracy | — | 0.7848 |
| F1 (macro) | **0.724 ± 0.066** | 0.6815 |
| F1 (minority class=1) | — | 0.5000 |
| AUC-ROC | **0.738** | 0.7562 |
| Threshold | 0.626 | — |
| Training time | — | 20.6 s |

---

## Deep Learning Method

### Pipeline

```
SelectKBest(f_classif, k=12) → StandardScaler → MLP([16]) → sigmoid
```

1. **SelectKBest(f_classif, k=12)** — Same ANOVA F-test selection as the traditional method. Fit on training fold only during CV to prevent leakage. Reduces input from 125 → 12 features.
2. **StandardScaler** — Applied after selection so scale normalisation matches the selected feature subset.
3. **MLP: 12 → 16 → 1** — Single hidden layer with BatchNorm, ReLU, and Dropout(0.25). Parameter count ≈ 220. The drastically small network is essential: with 158 samples, a 50k-parameter network (our v3) overfits catastrophically; a 220-parameter network does not.
4. **Sigmoid output** used with `BCEWithLogitsLoss(pos_weight ≈ 3.94)`.

**No SMOTE** — pos_weight handles the 4:1 imbalance in the loss function.

### Training

| Setting | Value | Rationale |
|---|---|---|
| Optimizer | AdamW (lr=1e-3, wd=0.01) | Decouples weight decay from gradient updates; stronger regularisation |
| Loss | BCEWithLogitsLoss(pos_weight=3.94) | Weights minority-class loss by original class ratio |
| Early stopping | patience=20 (max 150 epochs) | Restores best-validation-F1 checkpoint; avg stop ≈ 8 epochs |
| Final training | 250 epochs, 5-seed ensemble | No validation set; ensemble over 5 random initialisations |

### Config Sweep

Three architectures were evaluated using **RepeatedStratifiedKFold (5×3 = 15 folds)** with early stopping and per-fold threshold optimisation.

| Config | Architecture | CV F1 (macro) | CV AUC-ROC | Avg epoch |
|---|---|---|---|---|
| **k12_16** | 12→16→1 | **0.758 ± 0.085** | 0.706 | 7.9 |
| k8_16 | 8→16→1 | 0.745 ± 0.068 | 0.722 | 6.1 |
| k8_32_8 | 8→32→8→1 | 0.744 ± 0.069 | 0.705 | 9.0 |

### Results

| Metric | CV estimate | Training (full dev set) |
|---|---|---|
| Accuracy | — | 0.7658 |
| F1 (macro) | **0.758 ± 0.085** | 0.6823 |
| F1 (minority class=1) | — | 0.5195 |
| AUC-ROC | **0.706** | 0.7904 |
| Threshold | 0.496 | — |
| Training time | — | 268 s (5-seed ensemble) |

The threshold of **0.496 ≈ 0.5** indicates well-calibrated probabilities, a major improvement over earlier versions where poorly calibrated outputs required thresholds of 0.945.

---

## Comparison

| Metric | Traditional | Deep Learning |
|---|---|---|
| CV F1 (macro) | 0.724 ± 0.066 | **0.758 ± 0.085** |
| CV AUC-ROC | **0.738** | 0.706 |
| Training time | **20.6 s** | 268 s |
| Threshold | 0.626 | 0.496 |
| Key hyperparameter | k=10, C=0.1 | k=12, hidden=[16] |

The deep model achieves higher mean CV F1 (+0.034) but with greater fold-to-fold variance (std 0.085 vs 0.066), reflecting the inherent instability of neural networks on a 158-sample dataset. The traditional model has higher AUC-ROC (0.738 vs 0.706), indicating better probability ranking even if the thresholded F1 is slightly lower. Both methods are competitive; the traditional model is the more conservative and reproducible choice.
