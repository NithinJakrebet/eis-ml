import numpy as np
import pandas as pd
from scipy.stats import norm

import metrics


def _preds(scale_true=3.0, n_cells=6, n=200, seed=0):
    rng = np.random.RandomState(seed)
    frames = []
    for c in range(n_cells):
        std = rng.uniform(0.02, 0.05, n)
        resid = rng.normal(0, std * scale_true)  # sigma under-estimates the error by scale_true
        y_pred = rng.uniform(0.5, 1.0, n)
        frames.append(pd.DataFrame({
            "channel": f"C{c}", "cycle": np.arange(n), "y_true": y_pred + resid, "y_pred": y_pred, "y_std": std,
        }))
    return pd.concat(frames, ignore_index=True)


def test_conformal_scale_recovers_underestimation():
    preds = _preds(scale_true=3.0)
    scale = metrics.conformal_scale(preds, alpha=0.05)
    # 95% quantile of |N(0, 3)| is 1.96 * 3
    assert np.allclose(scale.median(), 1.96 * 3.0, rtol=0.1)
    cal = metrics.calibrate(preds, alpha=0.05)
    summary = metrics.conformal_summary(cal, alpha=0.05)
    assert abs(summary["coverage"] - 0.95) < 0.02
    assert summary["per_cell_coverage_min"] > 0.9
    # y_std_cal is a sigma whose z_{0.975} band equals the conformal interval
    assert np.allclose(norm.ppf(0.975) * cal["y_std_cal"], cal["sigma_scale"] * cal["y_std"])


def test_conformal_scale_is_leave_one_cell_out():
    preds = _preds()
    scale = metrics.conformal_scale(preds, alpha=0.1)
    # one value per cell, and it does not depend on the cell's own rows
    per_cell = scale.groupby(preds["channel"]).nunique()
    assert (per_cell == 1).all()
    tweaked = preds.copy()
    tweaked.loc[tweaked["channel"] == "C0", "y_true"] += 10.0  # huge residuals in C0 only
    scale2 = metrics.conformal_scale(tweaked, alpha=0.1)
    assert scale2[tweaked["channel"] == "C0"].iloc[0] == scale[preds["channel"] == "C0"].iloc[0]
    assert (scale2[tweaked["channel"] != "C0"] > scale[preds["channel"] != "C0"]).all()


def test_coverage_std_col():
    preds = metrics.calibrate(_preds(), alpha=0.05)
    raw = metrics.coverage(preds)
    cal = metrics.coverage(preds, std_col="y_std_cal")
    assert cal["within_2sigma"] > raw["within_2sigma"]
