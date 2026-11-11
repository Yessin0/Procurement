"""Where the trained model and its metrics live. MODEL_PATH comes from .env (default below)."""
import json
import os
from pathlib import Path

import joblib
from django.conf import settings

DEFAULT_MODEL_PATH = "ml_pipeline/artifacts/classifier.pkl"


def artifact_path() -> Path:
    path = Path(os.getenv("MODEL_PATH", DEFAULT_MODEL_PATH))
    return path if path.is_absolute() else Path(settings.BASE_DIR) / path


def metrics_path() -> Path:
    return artifact_path().with_name("metrics.json")


def save(model, threshold: float, version: str, feature_columns: list, metrics: dict) -> None:
    path = artifact_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": model, "threshold": threshold, "model_version": version,
                 "feature_columns": feature_columns}, path)
    metrics_path().write_text(json.dumps(metrics, indent=2))


def load() -> dict:
    """Returns {'model', 'threshold', 'model_version', 'feature_columns'}. Only load files you trained yourself."""
    path = artifact_path()
    if not path.exists():
        raise FileNotFoundError(f"No trained model at {path}. Run: python manage.py train_model")
    return joblib.load(path)


def load_metrics() -> dict:
    """For GET /api/metrics/model/ (written by train_model)."""
    path = metrics_path()
    if not path.exists():
        raise FileNotFoundError(f"No metrics at {path}. Run: python manage.py train_model")
    return json.loads(path.read_text())
