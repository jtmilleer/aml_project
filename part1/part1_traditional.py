import warnings
import joblib

import pandas as pd
import numpy as np
import os
import time
import inspect

from sklearn.model_selection import train_test_split, GridSearchCV, RandomizedSearchCV, StratifiedKFold, cross_val_score, cross_validate
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.svm import SVC
from lightgbm import LGBMClassifier

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from matplotlib.colors import LinearSegmentedColormap

warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")
warnings.filterwarnings("ignore", message="X does not have valid feature names")

TRAINING = True
DEV_PATH = "part1/PartI_dev.csv"
TEST_PATH = "part1/PartI_dev.csv"

# ---------------------------------------------------------------------------
# Model constructors
# ---------------------------------------------------------------------------

def build_lda(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("lda", LinearDiscriminantAnalysis()))
    return Pipeline(steps)


def build_qda(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("qda", QuadraticDiscriminantAnalysis()))
    return Pipeline(steps)


def build_logistic_regression(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("lr", LogisticRegression(random_state=42, max_iter=2000)))
    pipeline = Pipeline(steps)
    param_grid = {"lr__C": [0.01, 0.1, 1.0, 10.0]}
    return GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy")


def build_random_forest():
    return RandomForestClassifier(
        n_estimators=300,
        criterion="entropy",
        max_features="sqrt",
        min_samples_leaf=1,
        random_state=42,
        n_jobs=-1,
    )


def build_gradient_boosting():
    return HistGradientBoostingClassifier(
        max_iter=300,
        learning_rate=0.05,
        max_depth=6,
        l2_regularization=0.1,
        random_state=42,
    )


def build_knn(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("knn", KNeighborsClassifier(weights="distance")))
    pipeline = Pipeline(steps)
    param_grid = {"knn__n_neighbors": [k for k in range(1, 51) if k % 2 == 1]}
    return RandomizedSearchCV(pipeline, param_grid, n_iter=5, cv=3, n_jobs=-1, scoring="accuracy", random_state=42)


def build_svm(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    # Hardcode standard parameters to avoid O(N^2) explosion during GridSearch
    steps.append(("svm", SVC(kernel="rbf", C=1.0, gamma="scale", random_state=42)))
    return Pipeline(steps)

def build_lightgbm(scale=True):
    steps = [("scaler", StandardScaler())] if scale else []
    steps.append(("lgbm", LGBMClassifier(random_state=42, verbose=-1)))
    pipeline = Pipeline(steps)
    param_grid = {"lgbm__n_estimators": [100, 300, 500], "lgbm__learning_rate": [0.01, 0.1, 1]}
    return GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy")




# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, Y_train, Y_test, verbose=False):
    """Fit, predict, and report accuracy (+ full report if verbose)."""
    start_time = time.time()
    model.fit(X_train, Y_train)
    train_time = time.time() - start_time
    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds) * 100
    print(f"  {name:<35} {acc:.2f}% (train time: {train_time:.2f}s)")
    if verbose:
        print(classification_report(Y_test, preds))
    return model, acc


def cross_validate_model(name, model, X, Y, n_splits=5):
    """Stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    start_time = time.time()
    scoring = {'accuracy': 'accuracy', 'f1_macro': 'f1_macro'}
    scores = cross_validate(model, X, Y, cv=cv, scoring=scoring, n_jobs=-1)
    cv_time = time.time() - start_time
    
    return {
        "name": name,
        "acc_mean": scores['test_accuracy'].mean() * 100,
        "acc_std": scores['test_accuracy'].std() * 100,
        "f1_mean": scores['test_f1_macro'].mean(),
        "f1_std": scores['test_f1_macro'].std(),
        "time": cv_time
    }


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":

    if TRAINING:
        print("=" * 60)
        print("  PART I — TRADITIONAL METHOD — DEVELOPMENT & TRAINING")
        print("=" * 60)

        # ------------------------------------------------------------------
        # Load development set
        # ------------------------------------------------------------------
        df = pd.read_csv(DEV_PATH)
        Y = df["Y"].to_numpy()
        X = df[[f"X{i}" for i in range(1, 49)]].to_numpy()

        X_train, X_test, Y_train, Y_test = train_test_split(
            X, Y, test_size=0.2, random_state=42, stratify=Y
        )

        model_builders = {
            "Logistic Regression":    build_logistic_regression,
            "Random Forest":          build_random_forest,
            "Gradient Boosting":      build_gradient_boosting,
            "KNN":                    build_knn,
            "LDA":                    build_lda,
            "QDA":                    build_qda,
            "SVM":                    build_svm,
            "LightGBM":               build_lightgbm,
        }

        def make_pca_pipe(builder):
            sig = inspect.signature(builder)
            model = builder(scale=False) if 'scale' in sig.parameters else builder()
            return Pipeline([
                ("scaler", StandardScaler()),
                ("pca", PCA(n_components=0.95)),
                ("classifier", model)
            ])

        pca_builders = {f"{name} (PCA)": lambda b=builder: make_pca_pipe(b) 
                        for name, builder in model_builders.items() 
                        if name != "Logistic Regression"}

        all_builders = {**model_builders, **pca_builders}
        all_models = {name: builder() for name, builder in all_builders.items()}

        # ------------------------------------------------------------------
        # Section 1: Stratified 5-fold cross-validation
        # ------------------------------------------------------------------
        print()
        print("=" * 55)
        print("5-fold stratified CV — all models (mean ± std)")
        print("=" * 55)

        cv_results = []
        for name, model in all_models.items():
            print(f"  Training & Validating: {name:<25} ", end="", flush=True)
            res = cross_validate_model(name, model, X, Y)
            print(f"--> Done! ({res['time']:.1f}s)")
            cv_results.append(res)

        best_idx = max(range(len(cv_results)), key=lambda i: cv_results[i]['acc_mean'])
        best_model_name = cv_results[best_idx]['name']

        def fmt_col(val, width):
            return f"{val:^{width}}"

        header = f"| {fmt_col('Model', 25)} | {fmt_col('CV Accuracy', 19)} | {fmt_col('CV F1 (macro)', 17)} | {fmt_col('CV Time', 11)} |"
        sep = f"|{'-'*27}|{'-'*21}|{'-'*19}|{'-'*13}|"
        print(header)
        print(sep)
        for i, res in enumerate(cv_results):
            name = res['name']
            acc_str = f"{res['acc_mean']:.2f}% ± {res['acc_std']:.2f}%"
            f1_str = f"{res['f1_mean']:.3f} ± {res['f1_std']:.3f}"
            time_str = f"{res['time']:.2f}s"
            
            if i == best_idx:
                name = f"**{name}**"
                acc_str = f"**{acc_str}**"
                f1_str = f"**{f1_str}**"
                time_str = f"**{time_str}**"
                
            print(f"| {fmt_col(name, 25)} | {fmt_col(acc_str, 19)} | {fmt_col(f1_str, 17)} | {fmt_col(time_str, 11)} |")

        # ------------------------------------------------------------------
        # Plot CV Comparison
        # ------------------------------------------------------------------
        FIGURES_DIR = Path('Documentation/figures')
        FIGURES_DIR.mkdir(parents=True, exist_ok=True)
        
        # Sort results by accuracy and take top 5
        cv_results_sorted = sorted(cv_results, key=lambda x: x['acc_mean'], reverse=True)[:5]
        
        metrics  = ['F1 Macro', 'Accuracy']
        x        = np.arange(len(metrics))
        width    = 0.15
        colors   = ['#2979B8', '#E8632A', '#888888', '#2CA02C', '#D62728']

        fig, ax = plt.subplots(figsize=(8, 5))
        for i, (res, color) in enumerate(zip(cv_results_sorted, colors)):
            vals = [res['f1_mean'], res['acc_mean'] / 100.0]
            errs = [res['f1_std'], res['acc_std'] / 100.0]
            bars  = ax.bar(x + (i - 2) * width, vals, width,
                           color=color, alpha=0.85, label=res['name'], zorder=3)
            ax.errorbar(x + (i - 2) * width, vals, yerr=errs,
                        fmt='none', color='#333333', capsize=4, linewidth=1.5, zorder=4)
            for bar, v in zip(bars, vals):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.01,
                        f'{v:.3f}', ha='center', va='bottom', fontsize=8, fontweight='bold', rotation=90)

        ax.set_xticks(x); ax.set_xticklabels(metrics)
        ax.set_ylabel('Score')
        ax.set_ylim(0.50, min(1.0, max([r['acc_mean']/100 for r in cv_results_sorted]) + 0.15))
        ax.set_title('Top 5 Traditional Models Cross-Validated Performance\n(error bars = ±1 std across folds)', fontweight='bold')
        ax.legend(loc='lower left', bbox_to_anchor=(1, 0.5), fontsize=9)
        ax.yaxis.grid(True, alpha=0.35, zorder=0)
        ax.set_axisbelow(True)

        out = FIGURES_DIR / 'p1_trad_cv_comparison.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close()

        # ------------------------------------------------------------------
        # Section 2: Detailed report for best model
        # ------------------------------------------------------------------
        # instantiate classifier
        best_model = all_builders[best_model_name]()
        
        print()
        print("=" * 55)
        print(f"Detailed classification report — {type(best_model).__name__}")
        print("=" * 55)
        
        # train classifier
        best_model.fit(X_train, Y_train)
        
        # report performance metrics
        print(classification_report(Y_test, best_model.predict(X_test)))

        print("Confusion Matrix:")
        print(confusion_matrix(Y_test, best_model.predict(X_test)))

        # ------------------------------------------------------------------
        # Plot Confusion Matrix
        # ------------------------------------------------------------------
        cm = confusion_matrix(Y_test, best_model.predict(X_test))
        fig, ax = plt.subplots(figsize=(7, 6))
        cmap = LinearSegmentedColormap.from_list('custom', ['#ffffff', '#2979B8'], N=256)
        sns.heatmap(cm, annot=True, fmt='d', cmap=cmap, linewidths=0.5, linecolor='#cccccc', ax=ax, cbar=False, annot_kws={'size': 9}, xticklabels=range(1, 12), yticklabels=range(1, 12))
        ax.set_title(f'Traditional Model ({best_model_name})\nConfusion Matrix (20% Hold-out Set)', fontsize=12, fontweight='bold')
        ax.set_xlabel('Predicted label')
        ax.set_ylabel('True label')
        out = FIGURES_DIR / 'p1_trad_confusion_matrix.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close()
        
        # ------------------------------------------------------------------
        # Plot PCA Explained Variance
        # ------------------------------------------------------------------
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        pca = PCA()
        pca.fit(X_scaled)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        
        fig, ax = plt.subplots(figsize=(7, 4.5))
        ax.plot(range(1, len(cumulative_variance) + 1), cumulative_variance, marker='o', linestyle='-', color='#2979B8', markersize=4, label='Cumulative Explained Variance')
        ax.axhline(y=0.95, color='#E8632A', linestyle='--', linewidth=1.5, label='95% Variance Threshold')
        n_comp = np.argmax(cumulative_variance >= 0.95) + 1
        ax.axvline(x=n_comp, color='#888888', linestyle=':', linewidth=1.5)
        ax.plot(n_comp, cumulative_variance[n_comp-1], 'o', color='#E8632A', ms=8)
        ax.text(n_comp + 1, 0.90, f'n={n_comp} components', color='#444444', fontweight='bold')
        ax.set_xlabel('Number of Principal Components')
        ax.set_ylabel('Cumulative Explained Variance')
        ax.set_title('PCA Dimensionality Reduction — Part I', fontweight='bold')
        ax.legend(loc='lower right', fontsize=10)
        ax.set_ylim(0.0, 1.05)
        ax.set_xlim(0, 50)
        ax.grid(True, alpha=0.3)
        out = FIGURES_DIR / 'p1_pca_variance.png'
        fig.savefig(out, bbox_inches='tight')
        plt.close()

        # ------------------------------------------------------------------
        # Section 3: Final model training & save
        # ------------------------------------------------------------------
        print(f"Retraining best model ({best_model_name}) on full development set...")
        best_model = all_builders[best_model_name]()
        best_model.fit(X, Y)
        
        # save classifier/parameters
        joblib.dump(best_model, "part1/part1_traditional_model.pkl")
        print("Best model saved to part1/part1_traditional_model.pkl")

    else:
        print("=" * 60)
        print("  PART I — TRADITIONAL METHOD — INFERENCE")
        print("=" * 60)

        # load test set
        df = pd.read_csv(TEST_PATH)
        Y = df["Y"].to_numpy()
        X = df[[f"X{i}" for i in range(1, 49)]].to_numpy()

        # instantiate classifier / load classifier/parameters
        classifier = joblib.load("part1/part1_traditional_model.pkl")

        # run classifier on test set
        preds = classifier.predict(X)

        # report performance metrics
        acc = accuracy_score(Y, preds) * 100
        print(f"Test Set Accuracy: {acc:.2f}%\n")
        print("Test Set Classification Report:")
        print(classification_report(Y, preds))