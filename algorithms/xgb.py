"""XGBoost ensemble for SOH, following Jones et al. (2022).

``n_models`` regressors are each trained on a different random 80% of the
training rows with a different seed. The ensemble mean is the prediction and
the spread across members is the uncertainty. NaN features are handled
natively by the trees, so sparse EIS grids need no imputation.
"""

from __future__ import annotations

import numpy as np
from sklearn.model_selection import train_test_split
from xgboost import XGBRegressor

DEFAULT_PARAMS = {
    "n_estimators": 500,
    "max_depth": 8,
    "learning_rate": 0.1,
    "subsample": 0.8,
    "colsample_bytree": 0.8,
    "tree_method": "hist",
    "n_jobs": 4,
}


def fit(X, y, n_models=10, random_state=42, **params):
    """Train an ensemble; extra keyword arguments override ``DEFAULT_PARAMS``."""
    X = np.asarray(X, dtype=float)
    y = np.asarray(y, dtype=float)
    cfg = {**DEFAULT_PARAMS, **params}
    members = []
    for i in range(n_models):
        seed = random_state + i
        X_tr, _, y_tr, _ = train_test_split(X, y, test_size=0.2, random_state=seed)
        members.append(XGBRegressor(**cfg, random_state=seed).fit(X_tr, y_tr))
    return {"models": members}


def predict(bundle, X):
    """Ensemble mean and standard deviation."""
    X = np.asarray(X, dtype=float)
    preds = np.stack([m.predict(X) for m in bundle["models"]])
    return preds.mean(axis=0), preds.std(axis=0)


def feature_importance(bundle) -> np.ndarray:
    """Gain-based importance averaged over ensemble members."""
    return np.mean([m.feature_importances_ for m in bundle["models"]], axis=0)
