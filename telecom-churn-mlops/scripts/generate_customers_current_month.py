import pandas as pd
import numpy as np
from pathlib import Path

np.random.seed(7)

data_dir = Path(__file__).resolve().parent.parent / "data"
data_dir.mkdir(parents=True, exist_ok=True)

df = pd.read_csv(data_dir / "customer_history.csv")

current = df.drop(columns=["churn_next_month"]).copy()

current["support_tickets_30d"] = (current["support_tickets_30d"] + (np.random.rand(len(current)) < 0.08).astype(int)).clip(0, 10)
current["late_payments_6m"] = (current["late_payments_6m"] + (np.random.rand(len(current)) < 0.05).astype(int)).clip(0, 6)

out_path = data_dir / "customers_current_month.csv"
current.to_csv(out_path, index=False)
print(f"Created: {out_path}")