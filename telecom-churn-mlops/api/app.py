from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel
from typing import List, Dict, Any, Optional
import pandas as pd
import joblib
import os

FEATURES = [
    "tenure_months",
    "monthly_charges",
    "total_data_gb",
    "support_tickets_30d",
    "late_payments_6m",
]

MODEL_PATH = "model/churn_model.pkl"
CURRENT_DATA_PATH = "data/customers_current_month.csv"

app = FastAPI(title="Churn Prediction API", version="1.1")

class PredictRequest(BaseModel):
    instances: List[Dict[str, Any]]
    threshold: Optional[float] = 0.7  # default

@app.on_event("startup")
def load_assets():
    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model not found at {MODEL_PATH}")
    app.state.model = joblib.load(MODEL_PATH)

    # Carregamos o “mês atual” para servir /top-risk rapidamente
    if os.path.exists(CURRENT_DATA_PATH):
        df = pd.read_csv(CURRENT_DATA_PATH)
        app.state.current_df = df
    else:
        app.state.current_df = None

@app.get("/health")
def health():
    return {"status": "ok"}

def _validate_and_prepare_df(df: pd.DataFrame) -> pd.DataFrame:
    missing = [c for c in FEATURES if c not in df.columns]
    if missing:
        raise HTTPException(status_code=400, detail=f"Missing required fields: {missing}")

    df_result = df[FEATURES].copy()
    for col in FEATURES:
        df_result[col] = pd.to_numeric(df_result[col], errors="coerce")

    if df_result.isna().any().any():
        bad = df_result.columns[df_result.isna().any()].tolist()
        raise HTTPException(status_code=400, detail=f"Invalid numeric values in fields: {bad}")

    return df_result

@app.post("/predict")
def predict(req: PredictRequest):
    """
    Returns a list of predictions with probabilities and predicted classes.
    """
    if not req.instances:
        raise HTTPException(status_code=400, detail="Empty instances.")

    df = pd.DataFrame(req.instances)
    X = _validate_and_prepare_df(df)

    model = app.state.model
    probs = model.predict_proba(X)[:, 1]

    threshold = req.threshold if req.threshold is not None else 0.7
    if threshold < 0 or threshold > 1:
        raise HTTPException(status_code=400, detail="threshold must be between 0 and 1")

    classes = (probs >= threshold).astype(int)

    # If "customer_id" in df.columns, include it in the output
    customer_ids = df["customer_id"].tolist() if "customer_id" in df.columns else [None] * len(df)

    return {
        "threshold": threshold,
        "predictions": [
            {
                "customer_id": cid,
                "churn_probability": float(p),
                "predicted_class": int(c)
            }
            for cid, p, c in zip(customer_ids, probs, classes)
        ],
    }

@app.get("/predict/{customer_id}")
def predict_by_customer_id(customer_id: str, threshold: float = 0.7):
    """
    Returns the churn probability and predicted class for the given customer_id, using the customers_current_month.csv file.
    """
    df = app.state.current_df
    if df is None:
        raise HTTPException(status_code=500, detail=f"{CURRENT_DATA_PATH} not loaded. Ensure the file exists.")

    if "customer_id" not in df.columns:
        raise HTTPException(status_code=500, detail="customers_current_month.csv must include customer_id")

    row = df[df["customer_id"] == customer_id]
    if row.empty:
        raise HTTPException(status_code=404, detail=f"customer_id not found: {customer_id}")

    X = _validate_and_prepare_df(row)
    model = app.state.model
    p = float(model.predict_proba(X)[:, 1])
    c = int(p >= threshold)

    return {
        "customer_id": customer_id,
        "threshold": threshold,
        "churn_probability": p,
        "predicted_class": c
    }

@app.get("/top-risk")
def top_risk(
    top_k: int = Query(50, ge=1, le=10000),
    threshold: float = Query(0.7, ge=0.0, le=1.0),
    include_all: bool = Query(False)
):
    """
    Returns the top_k customers with highest risk (churn probability),
    filtering by threshold (high risk).
    - include_all=false -> returns only those above the threshold (high risk)
    - include_all=true -> always returns top_k, even if some are below the threshold
    """
    df = app.state.current_df
    if df is None:
        raise HTTPException(status_code=500, detail=f"{CURRENT_DATA_PATH} not loaded. Ensure the file exists.")

    if "customer_id" not in df.columns:
        raise HTTPException(status_code=500, detail="customers_current_month.csv must include customer_id")

    df_result = _validate_and_prepare_df(df)
    model = app.state.model
    probs = model.predict_proba(df_result)[:, 1]

    out = df[["customer_id"]].copy()
    out["churn_probability"] = probs

    out = out.sort_values("churn_probability", ascending=False)

    if not include_all:
        out = out[out["churn_probability"] >= threshold]

    out = out.head(top_k)

    return {
        "top_k_requested": top_k,
        "threshold": threshold,
        "returned": len(out),
        "customers": out.to_dict(orient="records"),
    }

@app.post("/reload-current-data")
def reload_current_data():
    """
    Reloads the customers_current_month.csv (useful if the file is updated).
    """
    if not os.path.exists(CURRENT_DATA_PATH):
        raise HTTPException(status_code=404, detail=f"File not found: {CURRENT_DATA_PATH}")

    app.state.current_df = pd.read_csv(CURRENT_DATA_PATH)
    return {"status": "reloaded", "rows": len(app.state.current_df)}