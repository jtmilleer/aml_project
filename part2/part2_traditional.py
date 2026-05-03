"""
Part II - Traditional Method (Improved)
Pipeline: SMOTE -> StandardScaler -> PCA -> best classifier
Adds: multi-model comparison, wider grid search, out-of-fold threshold optimization
"""

TRAINING = True  # Professor Beichel, set to False for testing

import time
import numpy as np
import pandas as pd
import joblib
from sklearn.base import clone
from sklearn.model_selection import StratifiedKFold, GridSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE
from imblearn.pipeline import Pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'part2/PartII_dev.csv'
TEST_DATA_PATH  = 'part2/PartII_dev.csv'   # Professor Beichel, replace this path
MODEL_PATH      = 'part2/part2_trad_model.joblib'

FEATURE_COLS = [f'X{i}' for i in range(1, 126)]
TARGET_COL   = 'Y'


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float64)
    y = df[TARGET_COL].values.astype(int)
    return X, y


def report_metrics(y_true, y_pred, y_prob=None, label=''):
    print(f'\n{"─"*50}')
    print(f'  {label}')
    print(f'{"─"*50}')
    print(f'  Accuracy       : {accuracy_score(y_true, y_pred):.4f}')
    print(f'  F1 (macro)     : {f1_score(y_true, y_pred, average="macro"):.4f}')
    print(f'  F1 (minority=1): {f1_score(y_true, y_pred, pos_label=1):.4f}')
    if y_prob is not None:
        print(f'  AUC-ROC        : {roc_auc_score(y_true, y_prob):.4f}')
    print()
    print(classification_report(y_true, y_pred, digits=4))
    print('  Confusion Matrix (rows=true, cols=pred):')
    print(confusion_matrix(y_true, y_pred))


def threshold_predict(model, X, threshold):
    probs = model.predict_proba(X)[:, 1]
    return (probs >= threshold).astype(int), probs


def find_optimal_threshold(model_template, X, y, cv):
    """Out-of-fold threshold search — avoids leaking threshold onto training data."""
    all_probs, all_true = [], []
    for tr_i, val_i in cv.split(X, y):
        m = clone(model_template)
        m.fit(X[tr_i], y[tr_i])
        all_probs.extend(m.predict_proba(X[val_i])[:, 1])
        all_true.extend(y[val_i])
    all_probs = np.array(all_probs)
    all_true  = np.array(all_true)
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(all_true, (all_probs >= t).astype(int), average='macro')
           for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    return best_t, max(f1s)


# ── Training Branch ────────────────────────────────────────────────────────────
if TRAINING:
    print('=' * 60)
    print('  PART II — TRADITIONAL (IMPROVED) — TRAINING')
    print('=' * 60)
    t_start = time.time()

    X, y = load_data(TRAIN_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')
    print(f'Imbalance   : {np.bincount(y)[1] / len(y):.1%} positive')

    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    smote = SMOTE(random_state=42, k_neighbors=4)

    # ── Phase 1: Model Selection ───────────────────────────────────────────────
    print('\n── Phase 1: Comparing classifiers (5-fold CV, F1 macro) ──')

    candidates = {
        'SVM + PCA': Pipeline([
            ('smote',  smote),
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=0.95, random_state=42)),
            ('clf',    SVC(kernel='rbf', probability=True,
                           class_weight='balanced', random_state=42)),
        ]),
        'LogReg + PCA': Pipeline([
            ('smote',  smote),
            ('scaler', StandardScaler()),
            ('pca',    PCA(n_components=0.95, random_state=42)),
            ('clf',    LogisticRegression(class_weight='balanced',
                                          max_iter=1000, random_state=42)),
        ]),
        'RandomForest': Pipeline([
            ('smote', smote),
            ('clf',   RandomForestClassifier(n_estimators=300,
                                              class_weight='balanced',
                                              random_state=42, n_jobs=-1)),
        ]),
        'HistGBM': Pipeline([
            ('smote',  smote),
            ('scaler', StandardScaler()),
            ('clf',    HistGradientBoostingClassifier(class_weight='balanced',
                                                      random_state=42,
                                                      max_iter=300,
                                                      learning_rate=0.05)),
        ]),
    }

    comparison = {}
    for name, pipe in candidates.items():
        scores = cross_val_score(pipe, X, y, cv=cv, scoring='f1_macro', n_jobs=1)
        comparison[name] = scores
        print(f'  {name:20s}: {scores.mean():.4f} ± {scores.std():.4f}')

    best_name = max(comparison, key=lambda k: comparison[k].mean())
    print(f'\n  Winner: {best_name}  (CV F1 = {comparison[best_name].mean():.4f})')

    # ── Phase 2: Grid Search on Winner ────────────────────────────────────────
    print(f'\n── Phase 2: GridSearchCV for {best_name} ──')

    param_grids = {
        'SVM + PCA': {
            'clf__C':     [0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1.0, 5.0],
            'clf__gamma': [0.001, 0.005, 0.01, 0.05, 0.1, 'scale', 'auto'],
        },
        'LogReg + PCA': {
            'clf__C':      [0.001, 0.01, 0.1, 1, 10],
            'clf__solver': ['lbfgs', 'saga'],
            'clf__penalty': ['l2'],
        },
        'RandomForest': {
            'clf__n_estimators':     [200, 500, 1000],
            'clf__max_depth':        [None, 10, 20],
            'clf__min_samples_leaf': [1, 2, 4],
            'clf__max_features':     ['sqrt', 'log2'],
        },
        'HistGBM': {
            'clf__max_iter':         [100, 200, 500],
            'clf__learning_rate':    [0.01, 0.05, 0.1],
            'clf__max_depth':        [3, 5, None],
            'clf__min_samples_leaf': [10, 20, 50],
        },
    }

    grid = GridSearchCV(
        candidates[best_name], param_grids[best_name],
        cv=cv, scoring='f1_macro', n_jobs=1, verbose=1,
        return_train_score=True, refit=True,
    )
    grid.fit(X, y)

    best_idx = grid.best_index_
    cv_res   = grid.cv_results_
    print(f'\nBest params   : {grid.best_params_}')
    print(f'CV F1 (macro) : {cv_res["mean_test_score"][best_idx]:.4f}'
          f' ± {cv_res["std_test_score"][best_idx]:.4f}')
    print(f'CV F1 (train) : {cv_res["mean_train_score"][best_idx]:.4f}'
          f' ± {cv_res["std_train_score"][best_idx]:.4f}')

    final_model = grid.best_estimator_

    # ── Phase 3: Threshold Optimisation via Out-of-Fold Probabilities ─────────
    print('\n── Phase 3: Threshold optimisation (out-of-fold) ──')
    optimal_threshold, oof_f1 = find_optimal_threshold(final_model, X, y, cv)
    print(f'  Optimal threshold: {optimal_threshold:.3f}  (OOF F1 = {oof_f1:.4f})')

    # ── Phase 4: Report Training Performance and Save ─────────────────────────
    y_pred_tr, y_prob_tr = threshold_predict(final_model, X, optimal_threshold)
    report_metrics(y, y_pred_tr, y_prob_tr,
                   f'Training Performance — {best_name} (threshold={optimal_threshold:.3f})')

    joblib.dump({'model': final_model, 'threshold': optimal_threshold}, MODEL_PATH)
    print(f'\nModel saved to: {MODEL_PATH}')
    print(f'Total time    : {time.time() - t_start:.1f}s')


# ── Inference Branch ───────────────────────────────────────────────────────────
else:
    print('=' * 60)
    print('  PART II — TRADITIONAL (IMPROVED) — INFERENCE')
    print('=' * 60)

    X, y = load_data(TEST_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')

    artifact  = joblib.load(MODEL_PATH)
    model     = artifact['model']
    threshold = artifact['threshold']
    print(f'Loaded threshold: {threshold:.3f}')

    y_pred, y_prob = threshold_predict(model, X, threshold)
    report_metrics(y, y_pred, y_prob, 'Test Set Performance')
