from xgboost import XGBClassifier

RANDOM_STATE = 42


def make_model() -> XGBClassifier:
    """The chosen estimator (see docs/eda_findings.md: tied with random forest, faster SHAP, smaller file).

    ~42% positives, so no imbalance handling. XGBoost handles NaN natively.
    """
    return XGBClassifier(
        n_estimators=300, max_depth=4, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8,
        eval_metric="logloss", random_state=RANDOM_STATE, n_jobs=-1,
    )
