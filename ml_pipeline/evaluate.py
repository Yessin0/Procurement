"""Group-aware validation. Contracts of one cartel are strongly correlated, so a cartel
is either entirely in train or entirely in test (GroupKFold by `cartel_id`)."""
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd
from sklearn.metrics import (average_precision_score, confusion_matrix, f1_score,
                             precision_recall_curve, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GroupKFold

from .features import FEATURE_COLUMNS, build_features
from .model import make_model

N_FOLDS = 5


@dataclass
class CVResult:
    folds: pd.DataFrame    # threshold-free metrics per fold
    oof_proba: np.ndarray  # out-of-fold probability for every row (same order as the frame)
    fold_models: list      # [(fitted model, positions of its test rows)]


def cross_validate(frame: pd.DataFrame, n_splits: int = N_FOLDS) -> CVResult:
    X = build_features(frame).reset_index(drop=True)
    y = frame["is_cartel"].astype(int).to_numpy()
    groups = frame["cartel_id"].to_numpy()

    oof = np.full(len(y), np.nan)
    rows, models = [], []
    for i, (train, test) in enumerate(GroupKFold(n_splits=n_splits).split(X, y, groups), start=1):
        model = make_model().fit(X.iloc[train], y[train])
        proba = model.predict_proba(X.iloc[test])[:, 1]
        oof[test] = proba
        rows.append({
            "fold": i, "n_test": len(test), "n_cases": len(np.unique(groups[test])),
            "pos_rate": float(y[test].mean()),
            "pr_auc": float(average_precision_score(y[test], proba)),
            "roc_auc": float(roc_auc_score(y[test], proba)) if len(np.unique(y[test])) == 2 else np.nan,
        })
        models.append((model, test))
    return CVResult(folds=pd.DataFrame(rows), oof_proba=oof, fold_models=models)


def choose_threshold(proba, flag_rate: float = 0.10) -> float:
    """Score above which `flag_rate` of all contracts are flagged (default: the riskiest 10%).

    An operating point set by analyst capacity. Maximising F1 does not work here: with ~42% positives and
    a modest model, F1 is highest when (almost) everything is flagged (precision ~0.42, recall ~1.0).
    """
    return float(np.quantile(np.asarray(proba, dtype=float), 1.0 - flag_rate))


def best_f1_threshold(y_true, proba) -> dict:
    """For transparency only: where F1 would put the threshold, and how much it flags."""
    precision, recall, thresholds = precision_recall_curve(y_true, proba)
    f1 = 2 * precision[:-1] * recall[:-1] / np.maximum(precision[:-1] + recall[:-1], 1e-12)
    best = int(np.argmax(f1))
    threshold = float(thresholds[best])
    return {"threshold": threshold, "f1": float(f1[best]),
            "flagged_share": float((np.asarray(proba) >= threshold).mean())}


def _none_if_nan(value):
    return None if value is None or (isinstance(value, float) and np.isnan(value)) else float(value)


def summarize(frame: pd.DataFrame, cv: CVResult, threshold: float, model_version: str,
              flag_rate: float = 0.10) -> dict:
    """Metrics in the shape of GET /api/metrics/model/ (README Section 6)."""
    y = frame["is_cartel"].astype(int).to_numpy()
    prec, rec, f1s = [], [], []
    for _, test in cv.fold_models:
        pred = (cv.oof_proba[test] >= threshold).astype(int)
        prec.append(precision_score(y[test], pred, zero_division=0))
        rec.append(recall_score(y[test], pred, zero_division=0))
        f1s.append(f1_score(y[test], pred, zero_division=0))

    pooled = (cv.oof_proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y, pooled, labels=[0, 1]).ravel()
    precision_pooled = tp / max(tp + fp, 1)
    folds = cv.folds
    return {
        "model_version": model_version,
        "evaluated_on": "group_kfold_by_cartel_id",
        "n_folds": int(len(folds)),
        "n_rows": int(len(y)),
        "threshold": float(threshold),
        "flag_rate": float(flag_rate),
        "baseline_pr_auc": round(float(y.mean()), 4),
        "pr_auc_mean": float(folds.pr_auc.mean()),
        "pr_auc_std": float(folds.pr_auc.std()),
        "roc_auc_mean": float(folds.roc_auc.mean()),
        "precision_mean": float(np.mean(prec)),
        "recall_mean": float(np.mean(rec)),
        "f1_mean": float(np.mean(f1s)),
        "confusion_matrix": {"tn": int(tn), "fp": int(fp), "fn": int(fn), "tp": int(tp)},
        "precision_pooled": float(precision_pooled),
        "recall_pooled": float(tp / max(tp + fn, 1)),
        "lift": float(precision_pooled / y.mean()) if y.mean() > 0 else None,
        "f1_optimal": best_f1_threshold(y, cv.oof_proba),
        "folds": [
            {"fold": int(r.fold), "n_test": int(r.n_test), "n_cases": int(r.n_cases),
             "pos_rate": float(r.pos_rate), "pr_auc": float(r.pr_auc), "roc_auc": _none_if_nan(r.roc_auc)}
            for r in folds.itertuples()
        ],
        "features": FEATURE_COLUMNS,
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }
