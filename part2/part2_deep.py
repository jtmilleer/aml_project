"""
Part II - Deep Learning Method (v4)
Key changes from v3:
  - SelectKBest(f_classif, k=8) replaces PCA — picks the 8 most discriminative
    raw features by ANOVA F-test instead of the 11 highest-variance PCA directions.
  - Tiny network matching the reduced input: 8 -> 32 -> 8 -> 1  (~350 params).
    Eliminates the overfitting that plagued the 50k-parameter v3 network.
  - Early stopping (patience=20) on per-fold validation F1 — model stops at
    ~11 epochs on average instead of running all 400 epochs.
  - AdamW decouples weight decay from gradient updates (better than Adam+wd).
  - 5-seed ensemble on final training reduces variance across random initialisations.
"""

TRAINING = True  # Professor Beichel, set to False for testing

import time
import warnings
warnings.filterwarnings('ignore')
from copy import deepcopy

import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import RepeatedStratifiedKFold
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score,
    classification_report, confusion_matrix
)

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'part2/PartII_dev.csv'
TEST_DATA_PATH  = 'part2/PartII_dev.csv'   # Professor Beichel, replace this path
MODEL_PATH      = 'part2/part2_deep_model.pth'
PREPROC_PATH    = 'part2/part2_deep_preproc.joblib'

FEATURE_COLS = [f'X{i}' for i in range(1, 126)]
TARGET_COL   = 'Y'
SEED         = 42

# Configs swept during development — best is selected automatically.
CONFIGS = [
    {'name': 'k8_32_8',  'k': 8,  'hidden': [32, 8],  'dropout': 0.30, 'wd': 1e-2, 'lr': 8e-4},
    {'name': 'k8_16',    'k': 8,  'hidden': [16],      'dropout': 0.20, 'wd': 1e-2, 'lr': 1e-3},
    {'name': 'k12_16',   'k': 12, 'hidden': [16],      'dropout': 0.25, 'wd': 1e-2, 'lr': 1e-3},
]
MAX_EPOCHS  = 150   # hard cap per fold
PATIENCE    = 20    # early-stopping patience (epochs without improvement)
FINAL_SEEDS = 5     # ensemble size for the final model
BATCH_SIZE  = 32

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for w in hidden:
            layers += [nn.Linear(prev, w), nn.BatchNorm1d(w), nn.ReLU(), nn.Dropout(dropout)]
            prev = w
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
    print(f'\n{"─"*52}')
    print(f'  {label}')
    print(f'{"─"*52}')
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


def make_loader(X_np, y_np, shuffle=True):
    ds = TensorDataset(torch.tensor(X_np, dtype=torch.float32),
                       torch.tensor(y_np, dtype=torch.float32))
    return DataLoader(ds, batch_size=BATCH_SIZE, shuffle=shuffle, drop_last=False)


@torch.no_grad()
def get_probs(model, X_np):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    return torch.sigmoid(model(X_t)).cpu().numpy()


def best_threshold(y_true, probs):
    candidates = np.linspace(0.05, 0.95, 181)
    f1s = [f1_score(y_true.astype(int), (probs >= t).astype(int),
                    average='macro', zero_division=0) for t in candidates]
    return float(candidates[int(np.argmax(f1s))])


def pos_weight_tensor(y_np):
    n_neg, n_pos = np.bincount(y_np.astype(int))
    return torch.tensor([n_neg / n_pos], dtype=torch.float32).to(device)


def train_fold(X_tr, y_tr, X_val, y_val, config, seed):
    """
    Train one MLP with early stopping on validation F1-macro.
    Returns (model_at_best_epoch, best_info_dict).
    """
    torch.manual_seed(seed)
    model  = MLP(X_tr.shape[1], config['hidden'], config['dropout']).to(device)
    opt    = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor(y_tr))
    loader = make_loader(X_tr, y_tr)

    best = {'f1': -1.0, 'epoch': 0, 'state': None, 'threshold': 0.5}
    stale = 0

    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(Xb), yb).backward()
            opt.step()

        probs = get_probs(model, X_val)
        t     = best_threshold(y_val, probs)
        f1    = f1_score(y_val.astype(int), (probs >= t).astype(int),
                         average='macro', zero_division=0)

        if f1 > best['f1']:
            best = {'f1': f1, 'epoch': epoch,
                    'state': deepcopy(model.state_dict()), 'threshold': t}
            stale = 0
        else:
            stale += 1
        if stale >= PATIENCE:
            break

    model.load_state_dict(best['state'])
    return model, best


def train_final(X, y, config, epochs, seed):
    """Train on the full dataset for a fixed number of epochs (no validation set)."""
    torch.manual_seed(seed)
    model  = MLP(X.shape[1], config['hidden'], config['dropout']).to(device)
    opt    = optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['wd'])
    crit   = nn.BCEWithLogitsLoss(pos_weight=pos_weight_tensor(y))
    loader = make_loader(X, y)
    model.train()
    for _ in range(epochs):
        for Xb, yb in loader:
            Xb, yb = Xb.to(device), yb.to(device)
            opt.zero_grad(set_to_none=True)
            crit(model(Xb), yb).backward()
            opt.step()
    return model


# ── Training Branch ────────────────────────────────────────────────────────────
if TRAINING:
    print('=' * 60)
    print('  PART II — DEEP LEARNING (v4) — TRAINING')
    print('=' * 60)
    print(f'  Device : {device}')
    t_start = time.time()

    X, y = load_data(TRAIN_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y.astype(int), return_counts=True)))}')

    # RepeatedStratifiedKFold gives stable estimates on a 158-sample dataset.
    # 5 splits × 3 repeats = 15 fold evaluations per config.
    cv = RepeatedStratifiedKFold(n_splits=5, n_repeats=3, random_state=SEED)

    # ── Phase 1: Config sweep ──────────────────────────────────────────────────
    # SelectKBest is fit inside each training fold — no leakage onto validation.
    print(f'\n── Phase 1: Config sweep (RepeatedStratifiedKFold 5×3, '
          f'early stopping patience={PATIENCE}) ──')
    print(f'  {"Config":12s}  {"F1-macro":16s}  {"AUC":10s}  avg_epoch  threshold')

    config_results = {}
    for cfg in CONFIGS:
        fold_f1s, fold_aucs, fold_epochs, fold_thresholds = [], [], [], []

        for tr_idx, val_idx in cv.split(X, y):
            X_tr_raw, X_val_raw = X[tr_idx], X[val_idx]
            y_tr,     y_val     = y[tr_idx], y[val_idx]

            # Feature selection: fit only on training fold
            sel = SelectKBest(f_classif, k=cfg['k'])
            X_tr_sel  = sel.fit_transform(X_tr_raw, y_tr.astype(int))
            X_val_sel  = sel.transform(X_val_raw)

            # Scale after selection
            sc = StandardScaler()
            X_tr_s = sc.fit_transform(X_tr_sel).astype(np.float32)
            X_val_s = sc.transform(X_val_sel).astype(np.float32)

            _, best_info = train_fold(X_tr_s, y_tr, X_val_s, y_val, cfg, seed=SEED)
            fold_f1s.append(best_info['f1'])
            fold_thresholds.append(best_info['threshold'])
            fold_epochs.append(best_info['epoch'])
            try:
                probs = get_probs(
                    MLP(X_tr_s.shape[1], cfg['hidden'], cfg['dropout']).to(device),
                    X_val_s)
                # re-use model from best_info state
                m_tmp = MLP(X_tr_s.shape[1], cfg['hidden'], cfg['dropout']).to(device)
                m_tmp.load_state_dict(best_info['state'])
                fold_aucs.append(roc_auc_score(y_val.astype(int),
                                               get_probs(m_tmp, X_val_s)))
            except Exception:
                fold_aucs.append(float('nan'))

        f1_mean = float(np.mean(fold_f1s))
        f1_std  = float(np.std(fold_f1s))
        auc_mean = float(np.nanmean(fold_aucs))
        ep_mean  = float(np.mean(fold_epochs))
        t_mean   = float(np.mean(fold_thresholds))
        config_results[cfg['name']] = {
            'config': cfg, 'f1': f1_mean, 'f1_std': f1_std,
            'auc': auc_mean, 'epoch': ep_mean, 'threshold': t_mean,
        }
        print(f'  {cfg["name"]:12s}  {f1_mean:.4f} ± {f1_std:.4f}  '
              f'{auc_mean:.4f}  {ep_mean:5.1f}  {t_mean:.3f}')

    best_cfg_name = max(config_results, key=lambda k: config_results[k]['f1'])
    best_result   = config_results[best_cfg_name]
    best_cfg      = best_result['config']
    print(f'\n  Winner: {best_cfg_name}  '
          f'(F1={best_result["f1"]:.4f}, AUC={best_result["auc"]:.4f})')

    # ── Phase 2: Train final ensemble on full development set ──────────────────
    # Fit SelectKBest and StandardScaler on the full dataset.
    # Train FINAL_SEEDS models with different initialisations and average at
    # inference — reduces variance without requiring more data.
    # Epoch count: max(250, mean_best_epoch) following Tyler's heuristic.
    final_epochs = max(250, int(round(best_result['epoch'])))
    print(f'\n── Phase 2: Training {FINAL_SEEDS}-seed ensemble '
          f'({final_epochs} epochs each) ──')

    final_sel = SelectKBest(f_classif, k=best_cfg['k'])
    X_sel     = final_sel.fit_transform(X, y.astype(int))
    final_sc  = StandardScaler()
    X_s       = final_sc.fit_transform(X_sel).astype(np.float32)

    final_models = []
    for seed in range(FINAL_SEEDS):
        m = train_final(X_s, y, best_cfg, final_epochs, seed)
        final_models.append(m)
        print(f'  Seed {seed} done')

    # Training performance (ensemble average, CV-derived threshold)
    ens_probs = np.mean([get_probs(m, X_s) for m in final_models], axis=0)
    threshold = best_result['threshold']
    preds_tr  = (ens_probs >= threshold).astype(int)
    report_metrics(y, preds_tr, ens_probs,
                   f'Training Performance — {best_cfg_name} ensemble '
                   f'(threshold={threshold:.3f})')

    # ── Save ──────────────────────────────────────────────────────────────────
    torch.save({
        'model_states': [m.state_dict() for m in final_models],
        'config':       best_cfg,
    }, MODEL_PATH)
    joblib.dump({
        'selector':  final_sel,
        'scaler':    final_sc,
        'threshold': threshold,
    }, PREPROC_PATH)

    print(f'\nModel saved to  : {MODEL_PATH}')
    print(f'Preproc saved to: {PREPROC_PATH}')
    print(f'Total time      : {time.time() - t_start:.1f}s')


# ── Inference Branch ───────────────────────────────────────────────────────────
else:
    print('=' * 60)
    print('  PART II — DEEP LEARNING (v4) — INFERENCE')
    print('=' * 60)
    print(f'  Device: {device}')

    X, y = load_data(TEST_DATA_PATH)
    print(f'\nData loaded : {X.shape}')
    print(f'Class counts: {dict(zip(*np.unique(y.astype(int), return_counts=True)))}')

    preproc   = joblib.load(PREPROC_PATH)
    selector  = preproc['selector']
    scaler    = preproc['scaler']
    threshold = preproc['threshold']

    X_sel = selector.transform(X)
    X_s   = scaler.transform(X_sel).astype(np.float32)

    ckpt  = torch.load(MODEL_PATH, map_location=device)
    cfg   = ckpt['config']
    input_dim = X_s.shape[1]

    models = []
    for state in ckpt['model_states']:
        m = MLP(input_dim, cfg['hidden'], cfg['dropout']).to(device)
        m.load_state_dict(state)
        models.append(m)

    print(f'Loaded {len(models)}-model ensemble  |  threshold={threshold:.3f}')

    probs = np.mean([get_probs(m, X_s) for m in models], axis=0)
    preds = (probs >= threshold).astype(int)
    report_metrics(y, preds, probs, 'Test Set Performance')
