"""Turn a frame of contracts into a frame of predictions (DataFrame in, DataFrame out)."""
import numpy as np
import pandas as pd

from .evaluate import CVResult
from .explain import top_features
from .features import build_features


def _assemble(frame, proba, top, threshold, version) -> pd.DataFrame:
    return pd.DataFrame({
        "contract_id": frame["contract_id"].to_numpy(),
        "risk_score": np.asarray(proba, dtype=float),
        "predicted_label": np.asarray(proba) >= threshold,
        "threshold": float(threshold),
        "top_features": top,
        "model_version": version,
    })


def score_final(frame: pd.DataFrame, model, threshold: float, version: str) -> pd.DataFrame:
    """Score with the final model (for new contracts the model has never seen)."""
    X = build_features(frame).reset_index(drop=True)
    return _assemble(frame, model.predict_proba(X)[:, 1], top_features(model, X), threshold, version)


def score_out_of_fold(frame: pd.DataFrame, cv: CVResult, threshold: float, version: str) -> pd.DataFrame:
    """Honest scores for the labelled historical dataset: every contract is scored (and explained)
    by a model that never saw its cartel. Scoring with the final model would be in-sample."""
    X = build_features(frame).reset_index(drop=True)
    top = [None] * len(X)
    for model, test in cv.fold_models:
        for pos, explanation in zip(test, top_features(model, X.iloc[test])):
            top[pos] = explanation
    return _assemble(frame, cv.oof_proba, top, threshold, version)
