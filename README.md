# Vertex Churn Pipeline

Small end-to-end example to generate synthetic telecom customer data, train a churn model, run batch inference and serve a prediction API.

**Project layout**
- `data/` — generated CSV datasets (ignored by git)
- `model/` — trained model artifacts (ignored by git)
- `scripts/` — data generation helpers
- `training/` — training and evaluation scripts
- `interface/` — simple batch prediction runner
- `api/` — FastAPI application

**Prerequisites**
- Python 3.9+ (recommended) and pip (or conda/micromamba)
- Recommended: create a virtual environment

Quick setup
```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Generate data
```bash
python3 ./scripts/generate_customer_history.py
python3 ./scripts/generate_customers_current_month.py
```

Train model
```bash
python3 ./training/train_model.py
```

Evaluate model
```bash
python3 ./training/evaluate_model.py
```

Batch predictions
```bash
python3 ./interface/batch_predict.py
```

Serve API (development)
```bash
uvicorn api.app:app --reload --port 8000
```

API
- POST `/predict` — body: JSON with `instances: [{...}, ...]` containing the features listed in `FEATURES` in [api/app.py](api/app.py#L1)

Postman collection
- A Postman collection for local testing is included at `postman/Telecom Churn - Local.postman_collection.json`.
- Import it into Postman and set the `hostname` variable to `http://127.0.0.1:8000` (or your server address), then run the `Health Check` and `Predict` requests.

