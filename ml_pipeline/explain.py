"""SHAP explanations: top features per contract, in the API shape [{feature, value, impact}]."""
import numpy as np
import pandas as pd


def _shap_values(model, X: pd.DataFrame) -> np.ndarray:
    try:
        import shap
        values = shap.TreeExplainer(model).shap_values(X)
        if isinstance(values, list):      # older versions: one array per class
            values = values[1]
        values = np.asarray(values)
        if values.ndim == 3:              # (rows, features, classes)
            values = values[:, :, 1]
        return values
    except Exception:
        # Same TreeSHAP values straight from XGBoost (last column is the bias term)
        import xgboost as xgb
        contribs = model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)
        return contribs[:, :-1]


def top_features(model, X: pd.DataFrame, k: int = 10) -> list:
    """For each row: the k features with the largest |SHAP value|.
    impact > 0 pushes toward 'suspicious', impact < 0 away from it."""
    shap_values = _shap_values(model, X)
    names = list(X.columns)
    raw = X.to_numpy(dtype=float)
    order = np.argsort(-np.abs(shap_values), axis=1)[:, :k]
    result = []
    for r in range(len(X)):
        result.append([
            {"feature": names[j],
             "value": None if np.isnan(raw[r, j]) else round(float(raw[r, j]), 6),
             "impact": round(float(shap_values[r, j]), 6)}
            for j in order[r]
        ])
    return result
