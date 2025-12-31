import pandas as pd
import joblib

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_data_gb",
    "support_tickets_30d",
    "late_payments_6m",
]

MODEL_PATH = "model/churn_model.pkl"
INPUT_PATH = "data/customers_current_month.csv"
OUTPUT_PATH = "data/churn_predictions.csv"

def main():
    model = joblib.load(MODEL_PATH)
    df = pd.read_csv(INPUT_PATH)

    # Make sure we have the required columns
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in input: {missing}")

    probs = model.predict_proba(df[FEATURES])[:, 1]

    out = df.copy()
    out["churn_probability"] = probs

    threshold = 0.7
    out["high_risk"] = (out["churn_probability"] >= threshold).astype(int)

    out.to_csv(OUTPUT_PATH, index=False)
    print(f"Predictions saved to: {OUTPUT_PATH}")
    print("Top 10 customers by risk:")
    cols_to_show = ["customer_id", "churn_probability", "high_risk"]
    cols_to_show = [c for c in cols_to_show if c in out.columns]
    print(out.sort_values("churn_probability", ascending=False)[cols_to_show].head(10))

if __name__ == "__main__":
    main()