import json
from pathlib import Path

import joblib
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score, classification_report, confusion_matrix
from sklearn.model_selection import train_test_split

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_data_gb",
    "support_tickets_30d",
    "late_payments_6m",
]

DATA_PATH = Path("data/customer_history.csv")
MODEL_PATH = Path("model/churn_model.pkl")
OUT_METRICS = Path("model/metrics.json")
OUT_TEST_PRED = Path("data/test_predictions.csv")

def main():
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Missing: {DATA_PATH}")
    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Missing: {MODEL_PATH} (train first)")

    df = pd.read_csv(DATA_PATH)

    # X/y with label (this is essential for "evaluation")
    X = df[FEATURES]
    y = df["churn_next_month"].astype(int)

    # Important: use the SAME split always (random_state)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = joblib.load(MODEL_PATH)

    # Probabilities of churn (class 1)
    proba = model.predict_proba(X_test)[:, 1]

    # Metrics independent of threshold
    roc_auc = roc_auc_score(y_test, proba)
    pr_auc = average_precision_score(y_test, proba)

    # MMetrics with threshold (e.g., 0.7)
    threshold = 0.7
    pred = (proba >= threshold).astype(int)

    report = classification_report(y_test, pred, output_dict=True)
    cm = confusion_matrix(y_test, pred).tolist()

    metrics = {
        "roc_auc": roc_auc,
        "pr_auc": pr_auc,
        "threshold": threshold,
        "classification_report": report,
        "confusion_matrix": cm,
        "n_test": int(len(y_test)),
        "positive_rate_test": float(y_test.mean()),
    }

    OUT_METRICS.parent.mkdir(exist_ok=True, parents=True)
    with open(OUT_METRICS, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)

    # Save test predictions
    test_out = X_test.copy()
    test_out["churn_next_month"] = y_test.values
    test_out["churn_probability"] = proba
    test_out["predicted_class"] = pred
    test_out.to_csv(OUT_TEST_PRED, index=False)

    print("✅ Evaluation complete")
    print(f"- ROC AUC: {roc_auc:.4f}")
    print(f"- PR  AUC: {pr_auc:.4f}")
    print(f"- Metrics saved to: {OUT_METRICS}")
    print(f"- Test predictions saved to: {OUT_TEST_PRED}")

if __name__ == "__main__":
    main()