"""
Generate report-ready figures for Part II v4 results.
Saves PNGs to Documentation/figures/  — include in LaTeX with \includegraphics.
Run from the project root: conda run -n aml_p2 python part2/generate_figures.py
"""

import warnings; warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import joblib
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from matplotlib.colors import LinearSegmentedColormap
import seaborn as sns
from pathlib import Path
from copy import deepcopy

from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    confusion_matrix, roc_curve, auc,
    f1_score, accuracy_score, balanced_accuracy_score, roc_auc_score
)
from torch.utils.data import TensorDataset, DataLoader
from sklearn.model_selection import StratifiedKFold, StratifiedShuffleSplit
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline

# ── Paths ──────────────────────────────────────────────────────────────────────
DATA_PATH     = 'part2/PartII_dev.csv'
TRAD_MODEL    = 'part2/part2_trad_model.joblib'
DEEP_MODEL    = 'part2/part2_deep_model.pth'
DEEP_PREPROC  = 'part2/part2_deep_preproc.joblib'
FIGURES_DIR   = Path('Documentation/figures')
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
FEATURE_COLS = [f'X{i}' for i in range(1, 126)]

# ── Style ──────────────────────────────────────────────────────────────────────
plt.rcParams.update({
    'font.family':   'DejaVu Sans',
    'font.size':     11,
    'axes.titlesize': 12,
    'axes.labelsize': 11,
    'figure.dpi':    150,
    'savefig.dpi':   200,
    'savefig.bbox':  'tight',
})
BLUE   = '#2979B8'
ORANGE = '#E8632A'
GREY   = '#888888'

# ── Load data ──────────────────────────────────────────────────────────────────
df = pd.read_csv(DATA_PATH)
X  = df[FEATURE_COLS].values.astype(np.float32)
y  = df['Y'].values.astype(int)

# ── Load traditional model ─────────────────────────────────────────────────────
trad_artifact = joblib.load(TRAD_MODEL)
trad_model    = trad_artifact['model']
trad_thresh   = trad_artifact['threshold']
trad_scores   = trad_model.predict_proba(X)[:, 1]
trad_preds    = (trad_scores >= trad_thresh).astype(int)

# ── Load deep model ────────────────────────────────────────────────────────────
class MLP(torch.nn.Module):
    def __init__(self, input_dim, hidden, dropout):
        super().__init__()
        layers, prev = [], input_dim
        for w in hidden:
            layers += [torch.nn.Linear(prev, w), torch.nn.BatchNorm1d(w),
                       torch.nn.ReLU(), torch.nn.Dropout(dropout)]
            prev = w
        layers.append(torch.nn.Linear(prev, 1))
        self.net = torch.nn.Sequential(*layers)
    def forward(self, x): return self.net(x).squeeze(1)

deep_preproc = joblib.load(DEEP_PREPROC)
deep_sel     = deep_preproc['selector']
deep_sc      = deep_preproc['scaler']
deep_thresh  = deep_preproc['threshold']

ckpt     = torch.load(DEEP_MODEL, map_location=device)
cfg      = ckpt['config']
X_sel    = deep_sel.transform(X)
X_s      = deep_sc.transform(X_sel).astype(np.float32)

deep_models = []
for state in ckpt['model_states']:
    m = MLP(X_s.shape[1], cfg['hidden'], cfg['dropout']).to(device)
    m.load_state_dict(state)
    m.eval()
    deep_models.append(m)

with torch.no_grad():
    X_t = torch.tensor(X_s, dtype=torch.float32).to(device)
    deep_scores = np.mean(
        [torch.sigmoid(m(X_t)).cpu().numpy() for m in deep_models], axis=0)
deep_preds = (deep_scores >= deep_thresh).astype(int)

print(f'Traditional — F1={f1_score(y,trad_preds,average="macro"):.4f}  AUC={roc_auc_score(y,trad_scores):.4f}')
print(f'Deep        — F1={f1_score(y,deep_preds,average="macro"):.4f}  AUC={roc_auc_score(y,deep_scores):.4f}')

# ── 80/20 Stratified Holdout — re-fit models on 80%, evaluate on held-out 20% ─
# The models saved to disk were trained on all 158 samples, so we re-fit fresh
# instances here to get uncontaminated holdout confusion matrices.
# CV-derived thresholds are reused as they are our best estimate.
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
tr_idx, ho_idx = next(sss.split(X, y))
X_tr, X_ho = X[tr_idx], X[ho_idx]
y_tr, y_ho = y[tr_idx], y[ho_idx]

# Traditional: re-fit LogReg + SelectKBest(k=10) on the 80% portion
trad_ho_pipe = make_pipeline(
    StandardScaler(),
    SelectKBest(f_classif, k=10),
    LogisticRegression(C=0.1, solver='liblinear', class_weight='balanced', random_state=42),
)
trad_ho_pipe.fit(X_tr, y_tr)
trad_ho_scores = trad_ho_pipe.predict_proba(X_ho)[:, 1]
trad_ho_preds  = (trad_ho_scores >= trad_thresh).astype(int)

# Deep: re-fit a single MLP (k=12, hidden=[16]) on the 80% portion
ho_sel   = SelectKBest(f_classif, k=12)
X_tr_sel = ho_sel.fit_transform(X_tr, y_tr)
X_ho_sel = ho_sel.transform(X_ho)
ho_sc    = StandardScaler()
X_tr_s2  = ho_sc.fit_transform(X_tr_sel).astype(np.float32)
X_ho_s2  = ho_sc.transform(X_ho_sel).astype(np.float32)

torch.manual_seed(42)
ho_mlp  = MLP(X_tr_s2.shape[1], [16], 0.25).to(device)
ho_opt  = torch.optim.AdamW(ho_mlp.parameters(), lr=1e-3, weight_decay=1e-2)
n_neg_ho, n_pos_ho = np.bincount(y_tr)
ho_crit = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor([n_neg_ho / n_pos_ho], dtype=torch.float32).to(device))
ho_loader = DataLoader(
    TensorDataset(torch.tensor(X_tr_s2), torch.tensor(y_tr.astype(np.float32))),
    batch_size=32, shuffle=True)
ho_mlp.train()
for _ in range(250):
    for Xb, yb in ho_loader:
        Xb, yb = Xb.to(device), yb.to(device)
        ho_opt.zero_grad(set_to_none=True)
        ho_crit(ho_mlp(Xb), yb).backward()
        ho_opt.step()
ho_mlp.eval()
with torch.no_grad():
    X_ho_t = torch.tensor(X_ho_s2).to(device)
    deep_ho_scores = torch.sigmoid(ho_mlp(X_ho_t)).cpu().numpy()
deep_ho_preds = (deep_ho_scores >= deep_thresh).astype(int)
print(f'Holdout Traditional — F1={f1_score(y_ho,trad_ho_preds,average="macro"):.4f}')
print(f'Holdout Deep        — F1={f1_score(y_ho,deep_ho_preds,average="macro"):.4f}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1a — Confusion Matrices (20% Stratified Holdout)
# ══════════════════════════════════════════════════════════════════════════════
def _cm_figure(y_true, preds_list, titles, colors, suptitle, filename):
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    fig.suptitle(suptitle, fontweight='bold', y=1.02)
    for ax, preds, title, color in zip(axes, preds_list, titles, colors):
        cm_data = confusion_matrix(y_true, preds)
        cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', color], N=256)
        sns.heatmap(
            cm_data, annot=True, fmt='d', cmap=cmap,
            linewidths=0.5, linecolor='#cccccc',
            xticklabels=['Pred 0', 'Pred 1'],
            yticklabels=['True 0', 'True 1'],
            ax=ax, cbar=False, annot_kws={'size': 14, 'weight': 'bold'},
        )
        ax.set_title(title, fontsize=11)
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        ax.text(0.5, -0.32,
                f'Accuracy={accuracy_score(y_true, preds):.3f}  '
                f'F1={f1_score(y_true, preds, average="macro"):.3f}',
                ha='center', transform=ax.transAxes, fontsize=9, color='#444444')
    plt.tight_layout()
    out = FIGURES_DIR / filename
    fig.savefig(out)
    plt.close()
    print(f'Saved: {out}')

_cm_figure(
    y_ho,
    [trad_ho_preds, deep_ho_preds],
    ['Traditional\n(LogReg + SelectKBest k=10)', 'Deep Learning\n(MLP + SelectKBest k=12)'],
    [BLUE, ORANGE],
    f'Confusion Matrices — 20% Stratified Holdout (n={len(y_ho)}) — Part II v4',
    'confusion_matrices.png',
)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 1b — Confusion Matrices (Full Training Set)
# ══════════════════════════════════════════════════════════════════════════════
_cm_figure(
    y,
    [trad_preds, deep_preds],
    ['Traditional\n(LogReg + SelectKBest k=10)', 'Deep Learning\n(MLP + SelectKBest k=12, ensemble)'],
    [BLUE, ORANGE],
    f'Confusion Matrices — Full Training Set (n={len(y)}) — Part II v4',
    'confusion_matrices_train.png',
)

# ══════════════════════════════════════════════════════════════════════════════
# Figure 2 — ROC Curves
# ══════════════════════════════════════════════════════════════════════════════
fig, ax = plt.subplots(figsize=(5.5, 5))

for scores, label, color in [
    (trad_scores, f'Traditional  (AUC = {roc_auc_score(y, trad_scores):.3f})', BLUE),
    (deep_scores, f'Deep Learning (AUC = {roc_auc_score(y, deep_scores):.3f})', ORANGE),
]:
    fpr, tpr, _ = roc_curve(y, scores)
    ax.plot(fpr, tpr, color=color, lw=2, label=label)

ax.plot([0, 1], [0, 1], '--', color=GREY, lw=1, label='Random (AUC = 0.500)')
ax.set_xlabel('False Positive Rate')
ax.set_ylabel('True Positive Rate')
ax.set_title('ROC Curves — Part II v4\n(Training Set)', fontweight='bold')
ax.legend(loc='lower right', fontsize=10)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.3)

out = FIGURES_DIR / 'roc_curves.png'
fig.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 3 — CV Performance Comparison (bar chart with error bars)
# ══════════════════════════════════════════════════════════════════════════════
# Values from the v4 training runs
cv_data = {
    'Traditional\n(LogReg+SKB k=10)': {
        'F1 Macro':          (0.724, 0.066),
        'AUC-ROC':           (0.738, None),
        'Balanced Accuracy': (None,  None),
    },
    'Deep Learning\n(MLP+SKB k=12)': {
        'F1 Macro':          (0.758, 0.085),
        'AUC-ROC':           (0.706, None),
        'Balanced Accuracy': (None,  None),
    },
}

metrics  = ['F1 Macro', 'AUC-ROC']
models   = list(cv_data.keys())
x        = np.arange(len(metrics))
width    = 0.30
colors   = [BLUE, ORANGE]

fig, ax = plt.subplots(figsize=(6.5, 4.5))
for i, (model, color) in enumerate(zip(models, colors)):
    vals  = [cv_data[model][m][0] for m in metrics]
    errs  = [cv_data[model][m][1] or 0 for m in metrics]
    bars  = ax.bar(x + (i - 0.5) * width, vals, width,
                   color=color, alpha=0.85, label=model, zorder=3)
    ax.errorbar(x + (i - 0.5) * width, vals, yerr=errs,
                fmt='none', color='#333333', capsize=4, linewidth=1.5, zorder=4)
    for bar, v in zip(bars, vals):
        ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                f'{v:.3f}', ha='center', va='bottom', fontsize=9, fontweight='bold')

ax.set_xticks(x); ax.set_xticklabels(metrics)
ax.set_ylabel('Score')
ax.set_ylim(0.55, 0.85)
ax.set_title('Cross-Validated Performance — Part II v4\n'
             '(error bars = ±1 std across folds)', fontweight='bold')
ax.legend(loc='lower right', fontsize=9)
ax.yaxis.grid(True, alpha=0.35, zorder=0)
ax.set_axisbelow(True)

out = FIGURES_DIR / 'cv_comparison.png'
fig.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 4 — Feature Selection: Top ANOVA F-scores
# ══════════════════════════════════════════════════════════════════════════════
sc_full  = StandardScaler()
X_scaled = sc_full.fit_transform(X)

sel_full = SelectKBest(f_classif, k='all')
sel_full.fit(X_scaled, y)

scores_all = sel_full.scores_
top_n = 20
top_idx    = np.argsort(scores_all)[::-1][:top_n]
top_feats  = [FEATURE_COLS[i] for i in top_idx]
top_scores = scores_all[top_idx]

# Highlight selected features (k=10 traditional, k=12 deep)
selected_10 = set(FEATURE_COLS[i] for i in np.argsort(scores_all)[::-1][:10])
selected_12 = set(FEATURE_COLS[i] for i in np.argsort(scores_all)[::-1][:12])

bar_colors = []
for f in top_feats:
    if f in selected_10:
        bar_colors.append(BLUE)       # in both k=10 and k=12
    elif f in selected_12:
        bar_colors.append(ORANGE)     # only in k=12
    else:
        bar_colors.append('#cccccc')  # not selected

fig, ax = plt.subplots(figsize=(8, 5))
bars = ax.barh(range(top_n), top_scores[::-1] if False else top_scores,
               color=bar_colors, edgecolor='white', linewidth=0.5)
ax.set_yticks(range(top_n))
ax.set_yticklabels(top_feats, fontsize=9)
ax.invert_yaxis()
ax.set_xlabel('ANOVA F-score')
ax.set_title(f'Top {top_n} Features by ANOVA F-score (SelectKBest)\n'
             f'Blue = selected by both models (k≤10)   '
             f'Orange = deep only (k=11–12)   Grey = not selected',
             fontweight='bold', fontsize=10)
ax.xaxis.grid(True, alpha=0.3)
ax.set_axisbelow(True)

# Legend patches
from matplotlib.patches import Patch
legend_elements = [
    Patch(color=BLUE,      label='Selected (traditional k=10 & deep k=12)'),
    Patch(color=ORANGE,    label='Selected (deep k=12 only)'),
    Patch(color='#cccccc', label='Not selected'),
]
ax.legend(handles=legend_elements, loc='lower right', fontsize=9)

out = FIGURES_DIR / 'feature_scores.png'
fig.savefig(out)
plt.close()
print(f'Saved: {out}')

# ══════════════════════════════════════════════════════════════════════════════
# Figure 5 — Precision-Recall Curves (better than ROC for imbalanced data)
# ══════════════════════════════════════════════════════════════════════════════
from sklearn.metrics import precision_recall_curve, average_precision_score

fig, ax = plt.subplots(figsize=(5.5, 5))
baseline = y.sum() / len(y)  # positive class prevalence

for scores, label, color, thresh in [
    (trad_scores, 'Traditional', BLUE,   trad_thresh),
    (deep_scores, 'Deep Learning', ORANGE, deep_thresh),
]:
    prec, rec, thresholds = precision_recall_curve(y, scores)
    ap = average_precision_score(y, scores)
    ax.plot(rec, prec, color=color, lw=2, label=f'{label}  (AP = {ap:.3f})')
    # Mark operating threshold
    idx = np.argmin(np.abs(thresholds - thresh))
    ax.plot(rec[idx], prec[idx], 'o', color=color, ms=8,
            markeredgecolor='white', markeredgewidth=1.5)

ax.axhline(baseline, color=GREY, lw=1, linestyle='--',
           label=f'Random baseline (AP = {baseline:.3f})')
ax.set_xlabel('Recall')
ax.set_ylabel('Precision')
ax.set_title('Precision-Recall Curves — Part II v4\n'
             '(dots = operating threshold, Training Set)', fontweight='bold')
ax.legend(loc='upper right', fontsize=10)
ax.set_xlim(-0.01, 1.01); ax.set_ylim(-0.01, 1.01)
ax.grid(True, alpha=0.3)

out = FIGURES_DIR / 'pr_curves.png'
fig.savefig(out)
plt.close()
