import pandas as pd
import numpy as np 
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis

# silences some warnings on my laptop
import warnings
warnings.filterwarnings("ignore", category=RuntimeWarning, module="threadpoolctl")

# csv has 11 classes, each equally balanced
def train_lda(X_train, Y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lda', LinearDiscriminantAnalysis())
    ])
    pipeline.fit(X_train, Y_train)
    return pipeline

def train_qda(X_train, Y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('qda', QuadraticDiscriminantAnalysis())
    ])
    pipeline.fit(X_train, Y_train)
    return pipeline

def train_random_forest(X_train, Y_train):
    rf = RandomForestClassifier(criterion='entropy',n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, Y_train)
    return rf

def train_logistic_regression(X_train, Y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('lr', LogisticRegression(random_state=42, max_iter=2000))
    ])
    pipeline.fit(X_train, Y_train)
    return pipeline

def train_gradient_boosting(X_train, Y_train):

    gb = HistGradientBoostingClassifier(max_iter=100, random_state=42)
    gb.fit(X_train, Y_train)
    return gb

def train_knn(X_train, Y_train):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95)),
        ('knn', KNeighborsClassifier(weights='distance'))
    ])

    # list of possible values for k
    param_grid = {'knn__n_neighbors': [x for x in range(1, 51) if x % 2 == 1]}
    print("Running GridSearchCV for KNN...")
    
    grid_search = GridSearchCV(pipeline, param_grid, cv=3, n_jobs=-1, scoring='accuracy')
    grid_search.fit(X_train, Y_train)
    print(f"Best n_neighbors found: {grid_search.best_params_['knn__n_neighbors']}")
    return grid_search.best_estimator_

def pca_analysis(X_train, X_test, Y_train, Y_test):
    pipeline = Pipeline([
        ('scaler', StandardScaler()),
        ('pca', PCA(n_components=0.95))
    ])
    pipeline.fit(X_train)
    X_train_pca = pipeline.transform(X_train)
    X_test_pca = pipeline.transform(X_test)
    print(f"Explained variance ratio: {pipeline.named_steps['pca'].explained_variance_ratio_.sum()}")
    print(f"Number of components: {pipeline.named_steps['pca'].n_components_}")
    return X_train_pca, X_test_pca, pipeline

if __name__ == "__main__":
    df = pd.read_csv("part1/PartI_dev.csv")
    Y = df['Y'].to_numpy()
    X = df[[f"X{i}" for i in range(1, 49)]].to_numpy()

    X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=42)
    
    # Reduces down to 31 components, accuracy goes up or down depending on model when using PCA data
    X_train_pca, X_test_pca, pca_pipeline = pca_analysis(X_train, X_test, Y_train, Y_test)

    #lr_model = train_logistic_regression(X_train, Y_train)
    #lr_pred = lr_model.predict(X_test)
    #print(f"Logistic Regression Accuracy: {accuracy_score(Y_test, lr_pred) * 100:.2f}%")

    rf_model = train_random_forest(X_train, Y_train)
    rf_pred = rf_model.predict(X_test)
    print(f"Random Forest Accuracy: {accuracy_score(Y_test, rf_pred) * 100:.2f}%\n")

    rf_pca_model = train_random_forest(X_train_pca, Y_train)
    rf_pca_pred = rf_pca_model.predict(X_test_pca)
    print(f"Random Forest PCA Accuracy: {accuracy_score(Y_test, rf_pca_pred) * 100:.2f}%\n")

    gb_model = train_gradient_boosting(X_train, Y_train)
    gb_pred = gb_model.predict(X_test)
    print(f"Gradient Boosting Accuracy: {accuracy_score(Y_test, gb_pred) * 100:.2f}%")
    
    gb_pca_model = train_gradient_boosting(X_train_pca, Y_train)
    gb_pca_pred = gb_pca_model.predict(X_test_pca)
    print(f"Gradient Boosting PCA Accuracy: {accuracy_score(Y_test, gb_pca_pred) * 100:.2f}%\n")
    
    knn_model = train_knn(X_train, Y_train)
    knn_pred = knn_model.predict(X_test)
    print(f"KNN Accuracy: {accuracy_score(Y_test, knn_pred) * 100:.2f}%")

    knn_pca_model = train_knn(X_train_pca, Y_train)
    knn_pca_pred = knn_pca_model.predict(X_test_pca)
    print(f"KNN PCA Accuracy: {accuracy_score(Y_test, knn_pca_pred) * 100:.2f}%")

    lda_model = train_lda(X_train, Y_train)
    lda_pred = lda_model.predict(X_test)
    print(f"LDA Accuracy: {accuracy_score(Y_test, lda_pred) * 100:.2f}%")
    
    lda_pca_model = train_lda(X_train_pca, Y_train)
    lda_pca_pred = lda_pca_model.predict(X_test_pca)
    print(f"LDA PCA Accuracy: {accuracy_score(Y_test, lda_pca_pred) * 100:.2f}%")
    
    qda_model = train_qda(X_train_pca, Y_train)
    qda_pred = qda_model.predict(X_test_pca)
    print(f"QDA Accuracy: {accuracy_score(Y_test, qda_pred) * 100:.2f}%")

    qda_pca_model = train_qda(X_train_pca, Y_train)
    qda_pca_pred = qda_pca_model.predict(X_test_pca)
    print(f"QDA PCA Accuracy: {accuracy_score(Y_test, qda_pca_pred) * 100:.2f}%")
    

    
