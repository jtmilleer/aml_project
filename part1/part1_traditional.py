import warnings
import joblib

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold, cross_val_score
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    VotingClassifier,
)
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

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
    return Pipeline(steps)


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
    return GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring="accuracy")


# ---------------------------------------------------------------------------
# PCA preprocessing
# ---------------------------------------------------------------------------

def make_pca_pipeline(variance_threshold=0.95):
    return Pipeline([
        ("scaler", StandardScaler()),
        ("pca", PCA(n_components=variance_threshold)),
    ])


def apply_pca(X_train, X_test, variance_threshold=0.95):
    pca_pipe = make_pca_pipeline(variance_threshold)
    pca_pipe.fit(X_train)
    n = pca_pipe.named_steps["pca"].n_components_
    var = pca_pipe.named_steps["pca"].explained_variance_ratio_.sum()
    print(f"PCA: {n} components retain {var*100:.2f}% variance")
    return pca_pipe.transform(X_train), pca_pipe.transform(X_test), pca_pipe


# ---------------------------------------------------------------------------
# Evaluation helpers
# ---------------------------------------------------------------------------

def evaluate_model(name, model, X_train, X_test, Y_train, Y_test, verbose=False):
    """Fit, predict, and report accuracy (+ full report if verbose)."""
    model.fit(X_train, Y_train)
    preds = model.predict(X_test)
    acc = accuracy_score(Y_test, preds) * 100
    print(f"  {name:<35} {acc:.2f}%")
    if verbose:
        print(classification_report(Y_test, preds))
    return model, acc


def cross_validate_model(name, model, X, Y, n_splits=5):
    """Stratified k-fold cross-validation."""
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    scores = cross_val_score(model, X, Y, cv=cv, scoring="accuracy", n_jobs=-1) * 100
    print(f"  {name:<35} {scores.mean():.2f}% ± {scores.std():.2f}%")
    return scores


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    # ------------------------------------------------------------------
    # Load data
    # ------------------------------------------------------------------
    df = pd.read_csv("part1/PartI_dev.csv")
    Y = df["Y"].to_numpy()
    X = df[[f"X{i}" for i in range(1, 49)]].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(
        X, Y, test_size=0.2, random_state=42, stratify=Y
    )

    X_train_pca, X_test_pca, pca_pipeline = apply_pca(X_train, X_test)
    print()

    # ------------------------------------------------------------------
    # Section 1: Hold-out accuracy — raw features
    # ------------------------------------------------------------------
    print("=" * 55)
    print("Hold-out accuracy — raw features")
    print("=" * 55)

    raw_models = {
        "Logistic Regression":    build_logistic_regression(),
        "Random Forest":          build_random_forest(),
        "Gradient Boosting":      build_gradient_boosting(),
        "KNN":                    build_knn(),
        "LDA":                    build_lda(),
        "QDA":                    build_qda(),
    }

    trained_raw = {}
    for name, model in raw_models.items():
        trained_model, _ = evaluate_model(name, model, X_train, X_test, Y_train, Y_test)
        trained_raw[name] = trained_model

    # ------------------------------------------------------------------
    # Section 2: Hold-out accuracy — PCA features
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Hold-out accuracy — PCA features")
    print("=" * 55)

    pca_models = {
        "Random Forest (PCA)":       build_random_forest(),
        "Gradient Boosting (PCA)":   build_gradient_boosting(),
        "KNN (PCA)":                 build_knn(scale=False),   # already scaled by PCA pipe
        "LDA (PCA)":                 build_lda(scale=False),
        "QDA (PCA)":                 build_qda(scale=False),
    }

    trained_pca = {}
    for name, model in pca_models.items():
        trained_model, _ = evaluate_model(name, model, X_train_pca, X_test_pca, Y_train, Y_test)
        trained_pca[name] = trained_model

    # ------------------------------------------------------------------
    # Section 3: Soft-voting ensemble (best raw models)
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Ensemble")
    print("=" * 55)

    # Re-build fresh estimators for the ensemble (VotingClassifier fits its own copies)
    ensemble = VotingClassifier(
        estimators=[
            ("rf",  build_random_forest()),
            ("gb",  build_gradient_boosting()),
            ("lda", build_lda()),
        ],
        voting="soft",
        n_jobs=-1,
    )
    evaluate_model("RF + GB + LDA (soft vote)", ensemble, X_train, X_test, Y_train, Y_test)

    # ------------------------------------------------------------------
    # Section 4: Stratified 5-fold cross-validation
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("5-fold stratified CV — raw features (mean ± std)")
    print("=" * 55)

    cv_models = {
        "Logistic Regression":  build_logistic_regression(),
        "Random Forest":        build_random_forest(),
        "Gradient Boosting":    build_gradient_boosting(),
        "LDA":                  build_lda(),
        "QDA":                  build_qda(),
    }

    for name, model in cv_models.items():
        cross_validate_model(name, model, X, Y)

    # ------------------------------------------------------------------
    # Section 5: Detailed report for best model
    # ------------------------------------------------------------------
    print()
    print("=" * 55)
    print("Detailed classification report — Gradient Boosting")
    print("=" * 55)
    gb_model = build_gradient_boosting()
    gb_model.fit(X_train, Y_train)
    print(classification_report(Y_test, gb_model.predict(X_test)))

    # ------------------------------------------------------------------
    # Section 6: Save best model
    # ------------------------------------------------------------------
    best_model = build_gradient_boosting()
    best_model.fit(X, Y)   # refit on full dataset before saving
    joblib.dump(best_model, "best_model.pkl")
    print("Best model saved to best_model.pkl")