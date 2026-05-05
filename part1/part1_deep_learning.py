"""
Part I - Deep Learning Method
"""

TRAINING = True  # Set to False for testing/inference

import time
import numpy as np
import pandas as pd
import joblib
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.metrics import accuracy_score, f1_score, classification_report, confusion_matrix

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

# ── Paths ──────────────────────────────────────────────────────────────────────
TRAIN_DATA_PATH = 'part1/PartI_dev.csv'
TEST_DATA_PATH  = 'part1/PartI_dev.csv'   # Replace this path for testing
MODEL_PATH      = 'part1/part1_deep_model.pth'
PREPROC_PATH    = 'part1/part1_deep_preproc.joblib'   # scaler + label_encoder

FEATURE_COLS = [f'X{i}' for i in range(1, 49)]
TARGET_COL   = 'Y'

# ── Hyperparameters ────────────────────────────────────────────────────────────
HIDDEN_DIMS  = [128, 64]
DROPOUT_RATE = 0.3
LR           = 1e-3
WEIGHT_DECAY = 1e-4
EPOCHS       = 200
BATCH_SIZE   = 64

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# ── Model ──────────────────────────────────────────────────────────────────────
class MLP(nn.Module):
    def __init__(self, input_dim, hidden_dims, num_classes, dropout):
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
        layers.append(nn.Linear(prev, num_classes))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x)


# ── Helpers ────────────────────────────────────────────────────────────────────
def load_data(path):
    df = pd.read_csv(path)
    X = df[FEATURE_COLS].values.astype(np.float32)
    y = df[TARGET_COL].values
    return X, y


def report_metrics(y_true, y_pred, label=''):
    print(f'\n{"─"*50}')
    print(f'  {label}')
    print(f'{"─"*50}')
    print(f'  Accuracy       : {accuracy_score(y_true, y_pred):.4f}')
    print(f'  F1 (macro)     : {f1_score(y_true, y_pred, average="macro"):.4f}')
    print()
    print(classification_report(y_true, y_pred, digits=4))
    print('  Confusion Matrix (rows=true, cols=pred):')
    print(confusion_matrix(y_true, y_pred))


def make_loader(X_np, y_np, batch_size, shuffle=True):
    ds = TensorDataset(
        torch.tensor(X_np, dtype=torch.float32),
        torch.tensor(y_np, dtype=torch.long),
    )
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle, drop_last=False)


def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0.0
    for Xb, yb in loader:
        Xb, yb = Xb.to(device), yb.to(device)
        optimizer.zero_grad()
        loss = criterion(model(Xb), yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * len(yb)
    return total_loss


@torch.no_grad()
def get_preds(model, X_np):
    model.eval()
    X_t = torch.tensor(X_np, dtype=torch.float32).to(device)
    logits = model(X_t)
    return torch.argmax(logits, dim=1).cpu().numpy()


def build_model(input_dim, num_classes):
    return MLP(input_dim, HIDDEN_DIMS, num_classes, DROPOUT_RATE).to(device)


# ── Training Branch ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    if TRAINING:
        print('=' * 60)
        print('  PART I — DEEP LEARNING — TRAINING')
        print('=' * 60)
        print(f'  Device: {device}')
        t_start = time.time()

        X, y_raw = load_data(TRAIN_DATA_PATH)
        
        # Encode labels to 0, 1, ..., num_classes-1
        label_encoder = LabelEncoder()
        y = label_encoder.fit_transform(y_raw)
        num_classes = len(label_encoder.classes_)
        
        print(f'\nData loaded : {X.shape}')
        print(f'Num classes : {num_classes}')
        print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')

        # ── 80/20 Train/Test Split ────────────────────────────────────────────────
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )

        # ── 5-fold CV for development metrics ─────────────────────────────────────
        kf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
        cv_accs, cv_f1s = [], []

        print(f'\nRunning 5-fold cross-validation ({EPOCHS} epochs / fold)...')
        cv_start_time = time.time()
        for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train, y_train)):
            X_tr, X_val = X_train[tr_idx], X_train[val_idx]
            y_tr, y_val = y_train[tr_idx], y_train[val_idx]

            # Fit scaler on training fold only
            scaler = StandardScaler()
            X_tr_s = scaler.fit_transform(X_tr)
            X_val_s = scaler.transform(X_val)

            loader = make_loader(X_tr_s, y_tr, BATCH_SIZE)

            model = build_model(X_tr_s.shape[1], num_classes)
            optimizer  = optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
            criterion  = nn.CrossEntropyLoss()
            scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                             optimizer, T_max=EPOCHS, eta_min=1e-6)

            for _ in range(EPOCHS):
                train_epoch(model, loader, optimizer, criterion)
                scheduler.step()

            preds = get_preds(model, X_val_s)

            acc = accuracy_score(y_val, preds)
            f1  = f1_score(y_val, preds, average='macro')
            cv_accs.append(acc)
            cv_f1s.append(f1)
            print(f'  Fold {fold+1}: Accuracy={acc:.4f}  F1(macro)={f1:.4f}')

        cv_time = time.time() - cv_start_time

        acc_mean = np.mean(cv_accs) * 100
        acc_std = np.std(cv_accs) * 100
        f1_mean = np.mean(cv_f1s)
        f1_std = np.std(cv_f1s)

        print()
        def fmt_col(val, width):
            return f"{val:^{width}}"

        header = f"| {fmt_col('Model', 25)} | {fmt_col('CV Accuracy', 19)} | {fmt_col('CV F1 (macro)', 17)} | {fmt_col('CV Time', 11)} |"
        sep = f"|{'-'*27}|{'-'*21}|{'-'*19}|{'-'*13}|"
        print(header)
        print(sep)
        
        name = "**MLP (Deep Learning)**"
        acc_str = f"**{acc_mean:.2f}% ± {acc_std:.2f}%**"
        f1_str = f"**{f1_mean:.3f} ± {f1_std:.3f}**"
        time_str = f"**{cv_time:.2f}s**"
        
        print(f"| {fmt_col(name, 25)} | {fmt_col(acc_str, 19)} | {fmt_col(f1_str, 17)} | {fmt_col(time_str, 11)} |")

        # ── Train best model on 80% to evaluate on 20% hold-out ───────────────────
        print('\nTraining model on 80% split to evaluate on 20% hold-out test set...')
        holdout_scaler = StandardScaler()
        X_train_s = holdout_scaler.fit_transform(X_train)
        X_test_s  = holdout_scaler.transform(X_test)
        
        holdout_loader = make_loader(X_train_s, y_train, BATCH_SIZE)
        holdout_model  = build_model(X_train_s.shape[1], num_classes)
        optimizer  = optim.Adam(holdout_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion  = nn.CrossEntropyLoss()
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=EPOCHS, eta_min=1e-6)
        
        for epoch in range(EPOCHS):
            train_epoch(holdout_model, holdout_loader, optimizer, criterion)
            scheduler.step()

        test_preds = get_preds(holdout_model, X_test_s)
        
        print('\n20% Hold-out Test Set Performance:')
        report_metrics(label_encoder.inverse_transform(y_test), 
                       label_encoder.inverse_transform(test_preds), 
                       'Hold-out Metrics')

        # ------------------------------------------------------------------
        # Plot Confusion Matrix (20% Hold-out)
        # ------------------------------------------------------------------
        FIGURES_DIR = Path('Documentation/figures')
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        cm = confusion_matrix(y_test, test_preds)
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#E8632A'], N=256)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, linewidths=0.5, linecolor='#cccccc', ax=ax, cbar=False, annot_kws={'size': 9}, xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
        ax.set_title('Deep Learning Model (MLP)\nConfusion Matrix (20% Hold-out Set)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        out = FIGURES_DIR / 'p1_deep_confusion_matrix.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close()

        # ── Final model on full development set ───────────────────────────────────
        print(f'\nTraining final model on full dev set ({EPOCHS} epochs)...')
        start_time = time.time()
        final_scaler = StandardScaler()
        X_s = final_scaler.fit_transform(X)

        final_loader = make_loader(X_s, y, BATCH_SIZE)

        final_model = build_model(X_s.shape[1], num_classes)
        optimizer  = optim.Adam(final_model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
        criterion  = nn.CrossEntropyLoss()
        scheduler  = optim.lr_scheduler.CosineAnnealingLR(
                         optimizer, T_max=EPOCHS, eta_min=1e-6)

        for epoch in range(EPOCHS):
            train_epoch(final_model, final_loader, optimizer, criterion)
            scheduler.step()
            if (epoch + 1) % 50 == 0:
                print(f'  Epoch {epoch+1:>4}/{EPOCHS}')

        retrain_time = time.time() - start_time
        print(f"Retraining completed in {retrain_time:.2f} seconds!")

        # Training performance
        preds_tr = get_preds(final_model, X_s)
        report_metrics(label_encoder.inverse_transform(y), 
                       label_encoder.inverse_transform(preds_tr),
                       'Training Performance (Full Dev Set)')

        # Save model weights + preprocessing artifacts
        torch.save({
            'model_state': final_model.state_dict(),
            'input_dim':   X_s.shape[1],
            'num_classes': num_classes,
        }, MODEL_PATH)
        
        joblib.dump({
            'scaler':        final_scaler,
            'label_encoder': label_encoder,
        }, PREPROC_PATH)

        print(f'\nModel saved to  : {MODEL_PATH}')
        print(f'Preproc saved to: {PREPROC_PATH}')
        print(f'Total time      : {time.time() - t_start:.1f}s')


    # ── Inference Branch ───────────────────────────────────────────────────────────
    else:
        print('=' * 60)
        print('  PART I — DEEP LEARNING — INFERENCE')
        print('=' * 60)
        print(f'  Device: {device}')

        X, y_raw = load_data(TEST_DATA_PATH)
        print(f'\nData loaded : {X.shape}')

        # Load preprocessing + model
        preproc       = joblib.load(PREPROC_PATH)
        scaler        = preproc['scaler']
        label_encoder = preproc['label_encoder']

        y = label_encoder.transform(y_raw)
        print(f'Class counts: {dict(zip(*np.unique(y, return_counts=True)))}')

        X_s = scaler.transform(X)

        ckpt  = torch.load(MODEL_PATH, map_location=device)
        model = build_model(ckpt['input_dim'], ckpt['num_classes'])
        model.load_state_dict(ckpt['model_state'])

        preds = get_preds(model, X_s)
        
        report_metrics(label_encoder.inverse_transform(y), 
                       label_encoder.inverse_transform(preds), 
                       'Test Set Performance')
