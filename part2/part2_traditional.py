"""
Part II - Traditional Method (v4)
"""

TRAINING = True  # Professor Beichel, set to False for testing

import time
import warnings
warnings.filterwarnings('ignore')

import numpy as np
import pandas as pd
import joblib
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.base import clone
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'part2/PartII_dev.csv'
TEST_DATA_PATH  = 'part2/PartII_dev.csv'   # Professor Beichel, replace this path
MODEL_PATH      = 'part2/part2_trad_model.joblib'

FEATURE_COLS = [f'X{i}' for i in range(1, 126)]
TARGET_COL   = 'Y'
SEED         = 42


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(path):
    """Load CSV from path and return feature matrix X (float32) and integer label vector y."""
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(int)
    return X, y


def report_metrics(y_true, y_pred, y_prob=None, label=''):
    """Print accuracy, macro/minority F1, AUC-ROC, classification report, and confusion matrix."""
    print(f'\n{"─"*52}')
    print(f'  {label}')
    print(f'{"─"*52}')
    print(f'  Accuracy       : {accuracy_score(y_true, y_pred):.4f}')
    print(f'  F1 (macro)     : {f1_score(y_true, y_pred, average="macro"):.4f}')
    print(f'  F1 (minority=1): {f1_score(y_true, y_pred, pos_label=1):.4f}')
    if y_prob is not None:
        print(f'  AUC-ROC        : {roc_auc_score(y_true, y_prob):.4f}')
    print()
    print(classification_report(y_true, y_pred, digits=4))
    print('  Confusion Matrix (rows=true, cols=pred):')
    print(confusion_matrix(y_true, y_pred))


def best_threshold(y_true, scores):
    """Per-fold threshold sweep — maximises F1 macro on the fold's validation set."""
    # Renamed from 'candidates' to 'thresholds' to avoid confusion with the
    # outer pipeline dict also named 'candidates'.
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true, scores >= t, average='macro', zero_division=0)
           for t in thresholds]
    return float(thresholds[int(np.argmax(f1s))])


def positive_scores(model, X):
    """Return class-1 probability scores for any sklearn classifier, regardless of API."""
    if hasattr(model, 'predict_proba'):
        return model.predict_proba(X)[:, 1]
    # Fallback for models that only expose decision_function (e.g. linear SVM)
    raw = model.decision_function(X)
    return 1.0 / (1.0 + np.exp(-raw))


def evaluate_model(pipe, X, y, cv):
    """
    Run CV, returning mean F1-macro, AUC, and the average per-fold threshold.
    Threshold is tuned per fold to avoid information leakage.
    """
    fold_f1s, fold_aucs, fold_thresholds = [], [], []
    for tr_idx, val_idx in cv.split(X, y):
        pipe_clone = clone(pipe)
        pipe_clone.fit(X[tr_idx], y[tr_idx])
        scores = positive_scores(pipe_clone, X[val_idx])
        t = best_threshold(y[val_idx], scores)
        fold_thresholds.append(t)
        preds = (scores >= t).astype(int)
        fold_f1s.append(f1_score(y[val_idx], preds, average='macro', zero_division=0))
        try:
            fold_aucs.append(roc_auc_score(y[val_idx], scores))
        except ValueError:
            fold_aucs.append(float('nan'))
    return (float(np.mean(fold_f1s)), float(np.std(fold_f1s)),
            float(np.nanmean(fold_aucs)), float(np.mean(fold_thresholds)))


# ── Training Branch ────────────────────────────────────────────────────────────
if TRAINING:
    print('=' * 60)
    print('  PART II — TRADITIONAL (v4) — TRAINING')
    print('=' * 60)
    t_start = time.time()

    X, y = load_data(TRAIN_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')
    print(f'Imbalance   : {np.bincount(y)[1] / len(y):.1%} positive')

    # RepeatedStratifiedKFold gives far more stable estimates than a single 5-fold
    # on only 158 samples.  5 splits × 5 repeats = 25 evaluations per model.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=5, random_state=SEED)

    # ── Phase 1: Model Comparison ──────────────────────────────────────────────
    # SelectKBest(f_classif) applies ANOVA F-test per feature — directly measures
    # how much each feature's distribution shifts between class 0 and class 1.
    print('\n── Phase 1: Model comparison (RepeatedStratifiedKFold 5×5) ──')
    print(f'  {"Model":38s}  F1-macro        AUC-ROC  Threshold')

    pipelines = {
        'LogReg + SelectKBest(k=7)': make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=7),
            LogisticRegression(C=0.1, solver='liblinear',
                               class_weight='balanced', random_state=SEED),
        ),
        'LogReg + SelectKBest(k=10)': make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=10),
            LogisticRegression(C=0.1, solver='liblinear',
                               class_weight='balanced', random_state=SEED),
        ),
        'SVM(RBF) + SelectKBest(k=8)': make_pipeline(
            StandardScaler(),
            SelectKBest(f_classif, k=8),
            SVC(C=1.0, kernel='rbf', class_weight='balanced',
                   probability=True, random_state=SEED),
        ),
        'LogReg + PCA(95%)  [v3 approach]': make_pipeline(
            StandardScaler(),
            PCA(n_components=0.95, random_state=SEED),
            LogisticRegression(C=0.1, solver='liblinear',
                               class_weight='balanced', random_state=SEED),
        ),
    }

    results = {}
    for name, pipe in pipelines.items():
        f1_mean, f1_std, auc_mean, thresh_mean = evaluate_model(pipe, X, y, cv)
        results[name] = (f1_mean, auc_mean, thresh_mean)
        print(f'  {name:38s}  {f1_mean:.4f} ± {f1_std:.4f}  {auc_mean:.4f}  {thresh_mean:.3f}')

    best_name = max(results, key=lambda k: results[k][0])
    best_f1, best_auc, best_threshold_val = results[best_name]
    print(f'\n  Winner: {best_name}')
    print(f'          F1={best_f1:.4f}  AUC={best_auc:.4f}  threshold={best_threshold_val:.3f}')

    # ── Phase 2: Train final model on full development set ─────────────────────
    print('\n── Phase 2: Training final model on full dev set ──')
    final_model = pipelines[best_name]
    final_model.fit(X, y)

    # ── Phase 3: Report and Save ───────────────────────────────────────────────
    y_scores = positive_scores(final_model, X)
    y_pred   = (y_scores >= best_threshold_val).astype(int)
    report_metrics(y, y_pred, y_scores,
                   f'{best_name}  (threshold={best_threshold_val:.3f})')

    joblib.dump({'model': final_model, 'threshold': best_threshold_val}, MODEL_PATH)
    print(f'\nModel saved to: {MODEL_PATH}')
    print(f'Total time    : {time.time() - t_start:.1f}s')


# ── Testing Branch ───────────────────────────────────────────────────────────
else:
    print('=' * 60)
    print('  PART II — TRADITIONAL (v4) — INFERENCE')
    print('=' * 60)

    X, y = load_data(TEST_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')

    artifact  = joblib.load(MODEL_PATH)
    model     = artifact['model']
    threshold = artifact['threshold']
    print(f'Loaded threshold: {threshold:.3f}')

    scores = positive_scores(model, X)
    y_pred = (scores >= threshold).astype(int)
    report_metrics(y, y_pred, scores, 'Test Set Performance')
