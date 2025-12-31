import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
from sklearn.ensemble import RandomForestClassifier

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_data_gb",
    "support_tickets_30d",
    "late_payments_6m",
]

DATA_PATH = "data/customer_history.csv"
MODEL_PATH = "model/churn_model.pkl"

def main():
    df = pd.read_csv(DATA_PATH)

    X = df[FEATURES]
    y = df["churn_next_month"].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = RandomForestClassifier(
        n_estimators=300,
        max_depth=8,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    preds = (proba >= 0.5).astype(int)

    print("ROC AUC:", roc_auc_score(y_test, proba))
    print(classification_report(y_test, preds))

    importances = pd.Series(model.feature_importances_, index=FEATURES).sort_values(ascending=False)
    print("\nFeature importance:")
    print(importances)

    joblib.dump(model, MODEL_PATH)
    print(f"\nModel saved to: {MODEL_PATH}")

if __name__ == "__main__":
    main()