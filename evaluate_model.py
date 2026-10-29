
from pathlib import Path

import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "Telco-Customer-Churn-cleaned.csv"
MODEL_DIR = BASE_DIR / "models"

# Load the same cleaned dataset and recreate the same test split
df = pd.read_csv(DATA_PATH)
X = df.drop(columns=["customerID", "Churn"])
y = df["Churn"]

_, X_test, _, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
    stratify=y,
)

models = {
    "Logistic Regression": "logistic_regression.joblib",
    "Default Decision Tree": "decision_tree_default.joblib",
    "Tuned Decision Tree": "decision_tree_tuned.joblib",
}

results = []

for name, filename in models.items():
    model = joblib.load(MODEL_DIR / filename)
    predictions = model.predict(X_test)

    results.append({
        "Model": name,
        "Accuracy": accuracy_score(y_test, predictions),
        "Churn Precision": precision_score(
            y_test, predictions, pos_label="Yes", zero_division=0
        ),
        "Churn Recall": recall_score(
            y_test, predictions, pos_label="Yes", zero_division=0
        ),
        "Churn F1": f1_score(
            y_test, predictions, pos_label="Yes", zero_division=0
        ),
    })

    print(f"\n--- {name} ---")
    print("Confusion Matrix (No, Yes):")
    print(confusion_matrix(y_test, predictions, labels=["No", "Yes"]))
    print(classification_report(
        y_test, predictions, labels=["No", "Yes"], zero_division=0
    ))

comparison = pd.DataFrame(results)

print("\n--- Final Test-Set Model Comparison ---")
print(comparison.round(4).to_string(index=False))
