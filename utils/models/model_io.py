import joblib
from pathlib import Path

MODEL_DIR = Path("trained_models")
MODEL_DIR.mkdir(exist_ok=True)

def save_model(model, name: str):
    joblib.dump(model, MODEL_DIR / f"{name}.joblib")

def load_model(name: str):
    return joblib.load(MODEL_DIR / f"{name}.joblib")