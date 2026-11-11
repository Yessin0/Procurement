from datetime import date

from .features import build_features
from .model import make_model


def new_model_version() -> str:
    return f"xgb_v1_{date.today().isoformat()}"


def fit_final(frame):
    """Fit the model on ALL labelled contracts (used to score new, unlabelled contracts)."""
    X = build_features(frame)
    y = frame["is_cartel"].astype(int).to_numpy()
    return make_model().fit(X, y)
