"""
Part II - Deep Learning Method (Improved)
Key changes from v1:
  - PCA preprocessing (125 -> ~11 features) to combat overfitting
  - pos_weight computed from ORIGINAL class ratio (not after SMOTE) so
    minority class is genuinely penalised even after SMOTE balances counts
  - CosineAnnealingLR for smoother convergence
  - Out-of-fold threshold optimisation instead of fixed 0.5
"""

TRAINING = True  # Professor Beichel, set to False for testing

import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)
from imblearn.over_sampling import SMOTE

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'part2/PartII_dev.csv'
TEST_DATA_PATH  = 'part2/PartII_dev.csv'   # Professor Beichel, replace this path
MODEL_PATH      = 'part2/part2_deep_model.pth'
PREPROC_PATH    = 'part2/part2_deep_preproc.joblib'   # scaler + PCA + threshold

FEATURE_COLS = [f'X{i}' for i in range(1, 126)]
TARGET_COL   = 'Y'

# ── Hyperparameters ────────────────────────────────────────────────────────────
PCA_VARIANCE = 0.95       # keep components explaining 95% of variance (~11 dims)
HIDDEN_DIMS  = [64, 32]   # smaller network matching compressed PCA input
DROPOUT_RATE = 0.4
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 400
BATCH_SIZE   = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for h in hidden_dims:
            layers += [
                nn.Linear(prev, h),
                nn.BatchNorm1d(h),
                nn.ReLU(),
                nn.Dropout(dropout),
            ]
            prev = h
        layers.append(nn.Linear(prev, 1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values.astype(np.float32)
    return X, y


def report_metrics(y_true, y_pred, y_prob=None, label=''):
    print(f'\n{"─"*50}')
    print(f'  {label}')
    print(f'{"─"*50}')
    yi, pi = y_true.astype(int), y_pred.astype(int)
    print(f'  Accuracy       : {accuracy_score(yi, pi):.4f}')
    print(f'  F1 (macro)     : {f1_score(yi, pi, average="macro"):.4f}')
    print(f'  F1 (minority=1): {f1_score(yi, pi, pos_label=1):.4f}')
    if y_prob is not None:
        print(f'  AUC-ROC        : {roc_auc_score(yi, y_prob):.4f}')
    print()
    print(classification_report(yi, pi, digits=4))
    print('  Confusion Matrix (rows=true, cols=pred):')
    print(confusion_matrix(yi, pi))


def make_loader(X_np, y_np, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.float32),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        criterion(model(Xb), yb).backward()
        optimizer.step()
        total += len(yb)
    return total


@torch.no_grad()
def get_probs(model, X_np):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    return torch.sigmoid(model(X_t)).cpu().numpy()


def find_optimal_threshold(all_true, all_probs):
    """Sweep thresholds on out-of-fold predictions to maximise F1 macro."""
    thresholds = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(all_true.astype(int), (all_probs >= t).astype(int), average='macro')
           for t in thresholds]
    best_t = thresholds[int(np.argmax(f1s))]
    return best_t, max(f1s)


def build_model(input_dim):
    return MLP(input_dim, HIDDEN_DIMS, DROPOUT_RATE).to(device)


# ── Training Branch ────────────────────────────────────────────────────────────
if TRAINING:
    print('=' * 60)
    print('  PART II — DEEP LEARNING (IMPROVED) — TRAINING')
    print('=' * 60)
    print(f'  Device: {device}')
    t_start = time.time()

    X, y = load_data(TRAIN_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y.astype(int), return_counts=True)))}')

    # pos_weight from ORIGINAL class ratio (before SMOTE)
    # This is critical: after SMOTE the ratio is ~1.0, so computing pos_weight
    # from the resampled data effectively disables it. We want it from the raw data.
    n_neg_orig, n_pos_orig = np.bincount(y.astype(int))
    orig_pos_weight = torch.tensor([n_neg_orig / n_pos_orig],
                                   dtype=torch.float32).to(device)
    print(f'pos_weight  : {orig_pos_weight.item():.2f}  (from original ratio)')

    # ── 5-fold CV for development metrics + out-of-fold threshold search ──────
    kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    cv_f1s, cv_aucs = [], []
    oof_probs, oof_true = np.zeros(len(y)), np.zeros(len(y))

    print(f'\nRunning 5-fold cross-validation ({EPOCHS} epochs / fold)...')
    for fold, (tr_idx, val_idx) in enumerate(kf.split(X, y)):
        X_tr, X_val = X[tr_idx], X[val_idx]
        y_tr, y_val = y[tr_idx], y[val_idx]

        # Fit scaler + PCA on training fold only (no leakage)
        scaler = StandardScaler()
        X_tr_s = scaler.fit_transform(X_tr)
        X_val_s = scaler.transform(X_val)

        pca = PCA(n_components=PCA_VARIANCE, random_state=42)
        X_tr_p = pca.fit_transform(X_tr_s)
        X_val_p = pca.transform(X_val_s)

        # SMOTE on the PCA-compressed training fold
        smote = SMOTE(random_state=42, k_neighbors=4)
        X_tr_sm, y_tr_sm = smote.fit_resample(X_tr_p, y_tr.astype(int))
        y_tr_sm = y_tr_sm.astype(np.float32)

        loader = make_loader(X_tr_sm, y_tr_sm, BATCH_SIZE)

        # pos_weight uses the ORIGINAL fold ratio, not the post-SMOTE ratio
        n_neg_f, n_pos_f = np.bincount(y_tr.astype(int))
        fold_pw = torch.tensor([n_neg_f / n_pos_f], dtype=torch.float32).to(device)

        model = build_model(X_tr_p.shape[1])
        optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion  = nn.BCEWithLogitsLoss(pos_weight=fold_pw)
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=EPOCHS, eta_min=1e-6)

        for _ in range(EPOCHS):
            train_epoch(model, loader, optimizer, criterion)
            scheduler.step()

        probs = get_probs(model, X_val_p)
        preds = (probs >= 0.5).astype(int)

        oof_probs[val_idx] = probs
        oof_true[val_idx]  = y_val

        f1  = f1_score(y_val.astype(int), preds, average='macro')
        auc = roc_auc_score(y_val.astype(int), probs)
        cv_f1s.append(f1)
        cv_aucs.append(auc)
        print(f'  Fold {fold+1}: F1={f1:.4f}  AUC={auc:.4f}'
              f'  (PCA kept {pca.n_components_} components)')

    print(f'\nCV F1  (macro): {np.mean(cv_f1s):.4f} ± {np.std(cv_f1s):.4f}')
    print(f'CV AUC-ROC    : {np.mean(cv_aucs):.4f} ± {np.std(cv_aucs):.4f}')

    # Out-of-fold threshold optimisation
    optimal_threshold, oof_f1 = find_optimal_threshold(oof_true, oof_probs)
    oof_preds = (oof_probs >= optimal_threshold).astype(int)
    oof_auc   = roc_auc_score(oof_true.astype(int), oof_probs)
    print(f'\nOOF threshold : {optimal_threshold:.3f}')
    print(f'OOF F1 (macro): {oof_f1:.4f}')
    print(f'OOF AUC-ROC   : {oof_auc:.4f}')

    # ── Final model on full development set ───────────────────────────────────
    print(f'\nTraining final model on full dev set ({EPOCHS} epochs)...')
    final_scaler = StandardScaler()
    X_s = final_scaler.fit_transform(X)

    final_pca = PCA(n_components=PCA_VARIANCE, random_state=42)
    X_p = final_pca.fit_transform(X_s)
    print(f'PCA: {X.shape[1]} features → {final_pca.n_components_} components')

    smote = SMOTE(random_state=42, k_neighbors=4)
    X_sm, y_sm = smote.fit_resample(X_p, y.astype(int))
    y_sm = y_sm.astype(np.float32)

    final_loader = make_loader(X_sm, y_sm, BATCH_SIZE)

    final_model = build_model(X_p.shape[1])
    optimizer  = optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=orig_pos_weight)
    scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                     optimizer, T_max=EPOCHS, eta_min=1e-6)

    for epoch in range(EPOCHS):
        train_epoch(final_model, final_loader, optimizer, criterion)
        scheduler.step()
        if (epoch + 1) % 100 == 0:
            print(f'  Epoch {epoch+1:>4}/{EPOCHS}')

    # Training performance on original (unaugmented, uncompressed) dev set
    probs_tr = get_probs(final_model, X_p)
    preds_tr = (probs_tr >= optimal_threshold).astype(int)
    report_metrics(y, preds_tr, probs_tr,
                   f'Training Performance (threshold={optimal_threshold:.3f})')

    # Save model weights + all preprocessing artifacts
    torch.save({
        'model_state': final_model.state_dict(),
        'input_dim':   final_pca.n_components_,
    }, MODEL_PATH)
    joblib.dump({
        'scaler':     final_scaler,
        'pca':        final_pca,
        'threshold':  optimal_threshold,
    }, PREPROC_PATH)

    print(f'\nModel saved to  : {MODEL_PATH}')
    print(f'Preproc saved to: {PREPROC_PATH}')
    print(f'Total time      : {time.time() - t_start:.1f}s')


# ── Inference Branch ───────────────────────────────────────────────────────────
else:
    print('=' * 60)
    print('  PART II — DEEP LEARNING (IMPROVED) — INFERENCE')
    print('=' * 60)
    print(f'  Device: {device}')

    X, y = load_data(TEST_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y.astype(int), return_counts=True)))}')

    # Load all preprocessing + model
    preproc   = joblib.load(PREPROC_PATH)
    scaler    = preproc['scaler']
    pca       = preproc['pca']
    threshold = preproc['threshold']
    print(f'Loaded threshold: {threshold:.3f}')

    X_s = scaler.transform(X)
    X_p = pca.transform(X_s)

    model = build_model(pca.n_components_)
    ckpt  = torch.load(MODEL_PATH, map_location=device)
    model.load_state_dict(ckpt['model_state'])

    probs = get_probs(model, X_p)
    preds = (probs >= threshold).astype(int)
    report_metrics(y, preds, probs, 'Test Set Performance')
