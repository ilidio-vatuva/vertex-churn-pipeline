import argparse
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).parent

DATA_DIR = ROOT / "data"
MODEL_DIR = ROOT / "model"

HISTORY_CSV = DATA_DIR / "customer_history.csv"
CURRENT_CSV = DATA_DIR / "customers_current_month.csv"
PRED_CSV = DATA_DIR / "churn_predictions.csv"

MODEL_FILE = MODEL_DIR / "churn_model.pkl"
METRICS_FILE = MODEL_DIR / "metrics.json"
TEST_PRED_CSV = DATA_DIR / "test_predictions.csv"

GEN_HISTORY_SCRIPT = ROOT / "scripts" / "generate_customer_history.py"
GEN_CURRENT_SCRIPT = ROOT / "scripts" / "generate_customers_current_month.py"
TRAIN_SCRIPT = ROOT / "training" / "train_model.py"
EVAL_SCRIPT = ROOT / "training" / "evaluate_model.py"
BATCH_SCRIPT = ROOT / "inference" / "batch_predict.py"


def run(cmd):
    print("\n>>", " ".join(map(str, cmd)))
    r = subprocess.run(cmd, check=False)
    if r.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {r.returncode}: {' '.join(map(str, cmd))}")

def ensure_dirs():
    DATA_DIR.mkdir(exist_ok=True)
    MODEL_DIR.mkdir(exist_ok=True)

def data_exists():
    return HISTORY_CSV.exists() and CURRENT_CSV.exists()

def generate_data(force=False):
    if data_exists() and not force:
        print(f"✅ Data already exists:\n- {HISTORY_CSV}\n- {CURRENT_CSV}\nSkipping generation.")
        return

    print("🧪 Generating data...")
    if not GEN_HISTORY_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {GEN_HISTORY_SCRIPT}")
    if not GEN_CURRENT_SCRIPT.exists():
        raise FileNotFoundError(f"Missing script: {GEN_CURRENT_SCRIPT}")

    run([sys.executable, str(GEN_HISTORY_SCRIPT)])
    run([sys.executable, str(GEN_CURRENT_SCRIPT)])

    if not data_exists():
        raise RuntimeError("Data generation finished but CSVs were not found. Check scripts output.")

def train(force=False):
    if MODEL_FILE.exists() and not force:
        print(f"✅ Model already exists: {MODEL_FILE}\nSkipping training.")
        return

    if not TRAIN_SCRIPT.exists():
        raise FileNotFoundError(f"Missing training script: {TRAIN_SCRIPT}")

    print("🏋️ Training model...")
    run([sys.executable, str(TRAIN_SCRIPT)])

    if not MODEL_FILE.exists():
        raise RuntimeError("Training finished but model file was not found. Check training script.")

def evaluate(force=False):
    """
    Avaliação = métricas usando histórico COM label (customer_history.csv).
    Cria:
      - model/metrics.json
      - data/test_predictions.csv
    """
    if METRICS_FILE.exists() and TEST_PRED_CSV.exists() and not force:
        print(f"✅ Evaluation already exists:\n- {METRICS_FILE}\n- {TEST_PRED_CSV}\nSkipping evaluation.")
        return

    if not EVAL_SCRIPT.exists():
        raise FileNotFoundError(f"Missing evaluation script: {EVAL_SCRIPT}")

    print("📊 Evaluating model...")
    run([sys.executable, str(EVAL_SCRIPT)])

    if not METRICS_FILE.exists():
        raise RuntimeError("Evaluation finished but metrics.json was not found. Check evaluate_model.py.")
    if not TEST_PRED_CSV.exists():
        raise RuntimeError("Evaluation finished but test_predictions.csv was not found. Check evaluate_model.py.")

def batch_predict(force=False):
    if PRED_CSV.exists() and not force:
        print(f"✅ Predictions already exist: {PRED_CSV}\nSkipping batch prediction.")
        return

    if not BATCH_SCRIPT.exists():
        raise FileNotFoundError(f"Missing batch predict script: {BATCH_SCRIPT}")

    print("📦 Running batch prediction...")
    run([sys.executable, str(BATCH_SCRIPT)])

    if not PRED_CSV.exists():
        raise RuntimeError("Batch prediction finished but output CSV was not found. Check batch_predict.py.")

def start_api(port=8000):
    print("🚀 Starting API...")
    run([sys.executable, "-m", "uvicorn", "api.app:app", "--reload", "--port", str(port)])

def main():
    parser = argparse.ArgumentParser(description="Local churn pipeline (Vertex-like).")
    parser.add_argument("--regen-data", action="store_true", help="Force data regeneration even if CSVs exist.")
    parser.add_argument("--retrain", action="store_true", help="Force model retraining even if model exists.")
    parser.add_argument("--reeval", action="store_true", help="Force evaluation even if metrics exist.")
    parser.add_argument("--repredict", action="store_true", help="Force batch prediction even if output exists.")
    parser.add_argument("--serve", action="store_true", help="Start the FastAPI server after pipeline.")
    parser.add_argument("--port", type=int, default=8000, help="Port for the API server (default: 8000).")
    args = parser.parse_args()

    ensure_dirs()
    generate_data(force=args.regen_data)
    train(force=args.retrain)
    evaluate(force=args.reeval)
    batch_predict(force=args.repredict)

    print("\n✅ Pipeline complete.")
    print(f"- data: {HISTORY_CSV} | {CURRENT_CSV}")
    print(f"- model: {MODEL_FILE}")
    print(f"- eval: {METRICS_FILE} | {TEST_PRED_CSV}")
    print(f"- predictions: {PRED_CSV}")

    if args.serve:
        start_api(port=args.port)

if __name__ == "__main__":
    main()