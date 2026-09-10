"""Evaluation of LOSO predictions.

All functions take the "predictions" table written by experiments: one row
per held-out (channel, cycle) with columns ``channel, cycle, y_true, y_pred,
y_std``. RMSE is the headline metric; on normalised SOH, per-cell R2 is
meaningless for cells that barely degraded (tiny variance), so the pooled
R2 over every held-out point is reported alongside the per-cell table.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

SOH_BANDS = ((0.0, 0.4), (0.4, 0.7), (0.7, 1.01))
_MIN_VAR = 1e-9


def scores(y_true, y_pred) -> dict[str, float]:
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    return {
        "rmse": float(np.sqrt(mean_squared_error(y_true, y_pred))),
        "mae": float(mean_absolute_error(y_true, y_pred)),
        "r2": float(r2_score(y_true, y_pred)) if y_true.var() > _MIN_VAR else float("nan"),
    }


def per_cell(preds: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for cell, g in preds.groupby("channel", sort=True):
        rows.append({
            "cell": cell,
            "n": len(g),
            "soh_std": float(g["y_true"].std()),
            **scores(g["y_true"], g["y_pred"]),
        })
    return pd.DataFrame(rows).sort_values("rmse", ascending=False).reset_index(drop=True)


def pooled(preds: pd.DataFrame) -> dict[str, float]:
    out = scores(preds["y_true"], preds["y_pred"])
    out["n"] = int(len(preds))
    return out


def by_soh_band(preds: pd.DataFrame, bands=SOH_BANDS) -> pd.DataFrame:
    """Where does the error live along the degradation axis?"""
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    rows = []
    for lo, hi in bands:
        m = (yt >= lo) & (yt < hi)
        if m.any():
            rows.append({
                "band": f"[{lo:.1f}, {hi:.1f})",
                "n": int(m.sum()),
                "rmse": float(np.sqrt(mean_squared_error(yt[m], yp[m]))),
                "bias": float((yp[m] - yt[m]).mean()),
            })
    return pd.DataFrame(rows)


def coverage(preds: pd.DataFrame, std_col: str = "y_std") -> dict[str, float]:
    """Fraction of residuals inside 1 and 2 predicted sigma (nominal 68% / 95%)."""
    z = (preds["y_true"] - preds["y_pred"]) / preds[std_col].clip(lower=1e-9)
    z = z.to_numpy()
    return {
        "within_1sigma": float(np.mean(np.abs(z) <= 1)),
        "within_2sigma": float(np.mean(np.abs(z) <= 2)),
        "mean_abs_z": float(np.mean(np.abs(z))),
        "mean_sigma": float(preds[std_col].mean()),
    }


def conformal_scale(preds: pd.DataFrame, alpha: float = 0.05) -> pd.Series:
    """Per-row multiplier that turns ``y_std`` into a (1-alpha) interval half-width.

    Split-conformal calibration, leave-one-cell-out: for the rows of cell c the
    scale is the finite-sample (1-alpha) quantile of |residual| / sigma over
    the *other* cells' held-out predictions, so the held-out cell never
    calibrates itself. Interval: ``y_pred +/- scale * y_std``.
    """
    z = ((preds["y_true"] - preds["y_pred"]).abs() / preds["y_std"].clip(lower=1e-9)).to_numpy()
    ch = preds["channel"].to_numpy()
    scale = np.empty(len(preds))
    for cell in pd.unique(ch):
        m = ch == cell
        others = z[~m]
        n = len(others)
        q = min(1.0, np.ceil((n + 1) * (1 - alpha)) / n)
        scale[m] = np.quantile(others, q)
    return pd.Series(scale, index=preds.index, name="sigma_scale")


def calibrate(preds: pd.DataFrame, alpha: float = 0.05) -> pd.DataFrame:
    """Add ``sigma_scale`` and ``y_std_cal`` (= scale * y_std / z_{1-alpha/2}) columns.

    ``y_std_cal`` is a sigma whose +/-1.96 (for alpha=0.05) band is the conformal
    interval, so the existing coverage / calibration plots apply unchanged.
    """
    from scipy.stats import norm

    out = preds.copy()
    out["sigma_scale"] = conformal_scale(out, alpha)
    out["y_std_cal"] = out["y_std"] * out["sigma_scale"] / norm.ppf(1 - alpha / 2)
    return out


def conformal_summary(preds: pd.DataFrame, alpha: float = 0.05) -> dict[str, float]:
    """Empirical coverage and width of the conformal interval (needs ``calibrate`` columns)."""
    half = preds["sigma_scale"] * preds["y_std"]
    inside = (preds["y_true"] - preds["y_pred"]).abs() <= half
    return {
        "alpha": alpha,
        "target_coverage": 1 - alpha,
        "coverage": float(inside.mean()),
        "median_sigma_scale": float(preds["sigma_scale"].median()),
        "mean_half_width": float(half.mean()),
        "per_cell_coverage_min": float(inside.groupby(preds["channel"]).mean().min()),
    }
