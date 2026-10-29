
from pathlib import Path

import joblib
import pandas as pd

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import (
    train_test_split,
    GridSearchCV,
    StratifiedKFold,
    cross_val_score,
)
from sklearn.metrics import make_scorer, f1_score


# 1. Project paths
BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn-cleaned.csv"
MODEL_DIR = BASE_DIR / "models"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# 2. Load cleaned dataset
df = pd.read_csv(DATA_PATH)

X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

print("Dataset shape:", df.shape)
print("Features shape:", X.shape)
print("\nTarget distribution:")
print(y.value_counts())


# 3. Train/test split
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

print("\nTraining rows:", len(X_train))
print("Testing rows:", len(X_test))


# 4. Preprocessing
numerical_features = X.select_dtypes(
    include=["int64", "float64"]
).columns.tolist()

categorical_features = X.select_dtypes(
    include=["object", "str", "string", "category"]
).columns.tolist()

preprocessor = ColumnTransformer(
    transformers=[
        (
            "categorical",
            OneHotEncoder(handle_unknown="ignore"),
            categorical_features,
        ),
        (
            "numerical",
            StandardScaler(),
            numerical_features,
        ),
    ]
)


# 5. Cross-validation setup
cv = StratifiedKFold(
    n_splits=5,
    shuffle=True,
    random_state=42,
)

churn_f1_scorer = make_scorer(
    f1_score,
    pos_label="Yes",
)


# 6. Train Logistic Regression
lr_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", LogisticRegression(max_iter=1000)),
    ]
)

lr_model.fit(X_train, y_train)

lr_cv_scores = cross_val_score(
    lr_model,
    X_train,
    y_train,
    cv=cv,
    scoring=churn_f1_scorer,
    n_jobs=1,
    error_score="raise",
)

print("\n--- Logistic Regression ---")
print("CV F1 scores:", lr_cv_scores.round(4))
print("Mean CV F1:", round(lr_cv_scores.mean(), 4))
print("CV standard deviation:", round(lr_cv_scores.std(), 4))

joblib.dump(
    lr_model,
    MODEL_DIR / "logistic_regression.joblib",
)


# 7. Train default Decision Tree
tree_model = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

tree_model.fit(X_train, y_train)

joblib.dump(
    tree_model,
    MODEL_DIR / "decision_tree_default.joblib",
)


# 8. Tune Decision Tree using training data only
param_grid = {
    "classifier__max_depth": [3, 5, 7, 10, None],
    "classifier__min_samples_leaf": [1, 5, 10, 20],
}

tuned_tree = Pipeline(
    steps=[
        ("preprocessor", clone(preprocessor)),
        ("classifier", DecisionTreeClassifier(random_state=42)),
    ]
)

search = GridSearchCV(
    estimator=tuned_tree,
    param_grid=param_grid,
    scoring=churn_f1_scorer,
    cv=cv,
    n_jobs=-1,
)

search.fit(X_train, y_train)

best_tree = search.best_estimator_

joblib.dump(
    best_tree,
    MODEL_DIR / "decision_tree_tuned.joblib",
)

print("\n--- Tuned Decision Tree ---")
print("Best settings:", search.best_params_)
print("Best CV F1:", round(search.best_score_, 4))

print("\nTraining complete.")
print("Saved models in:", MODEL_DIR)
