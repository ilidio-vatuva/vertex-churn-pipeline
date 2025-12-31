import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from pathlib import Path

np.random.seed(42)
n_customers = 20000

df = pd.DataFrame({
    "customer_id": [f"C{100000+i}" for i in range(n_customers)],
    "tenure_months": np.clip((np.random.gamma(2.0, 12.0, n_customers).astype(int) + 1), 1, 72),
})

df["monthly_charges"] = np.clip(np.random.normal(70, 20, n_customers), 20, 150).round(2)

base_data = np.random.lognormal(mean=3.5, sigma=0.6, size=n_customers)
df["total_data_gb"] = np.clip(base_data + (df["monthly_charges"] - 50) * 0.6, 0, 400).round(2)


lam_tickets = 0.6 + (df["monthly_charges"] > 90) * 0.8 + (df["tenure_months"] < 6) * 0.7
df["support_tickets_30d"] = np.random.poisson(np.clip(lam_tickets, 0.1, 5.0)).clip(0, 10)

lam_late = 0.4 + (df["monthly_charges"] > 90) * 0.5 + (df["support_tickets_30d"] >= 2) * 0.4
df["late_payments_6m"] = np.random.poisson(np.clip(lam_late, 0.1, 4.0)).clip(0, 6)

start = datetime.now(timezone.utc) - timedelta(days=7)
df["feature_time"] = [
    start + timedelta(seconds=int(s))
    for s in np.random.uniform(0, 7*24*3600, size=n_customers)
]

logit = (
    -2.5
    + 0.55 * df["support_tickets_30d"]
    + 0.65 * df["late_payments_6m"]
    - 0.03 * df["tenure_months"]
    + 0.01 * (df["monthly_charges"] - 70)
    + 0.002 * (df["total_data_gb"] - 80)
)
prob = 1 / (1 + np.exp(-logit))
df["churn_next_month"] = (np.random.rand(n_customers) < prob).astype(int)

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)
out_path = data_dir / "customer_history.csv"
df.to_csv(out_path, index=False)
print(f"Created: {out_path} | churn rate: {df['churn_next_month'].mean()}")