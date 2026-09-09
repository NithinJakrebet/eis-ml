"""Multi-step SOH forecasting with the horizon as the "action" (Jones et al. Fig. 4).

Our cells all follow one constant protocol, so the future charge/discharge
programme cannot be an input the way it is in Jones et al. The analogue is
the forecast horizon: predict SOH at cycle n+h from the EIS sweep at cycle n
plus the number of cycles ahead. One XGBoost model is trained on all
horizons at once; results are reported per horizon under LOSO.

    python experiments/xgb_multistep.py --dataset PEIS-HC-RT [--ns 6] [--horizons 0 1 5 10 20 40]
"""

from __future__ import annotations

import argparse
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from xgboost import XGBRegressor

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from eis_ml import build_dataset, get_dataset, load_dataset, loso_folds  # noqa: E402
from eis_ml import metrics, plots  # noqa: E402

warnings.filterwarnings("ignore")

PARAMS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": 4, "random_state": 42}


def make_pairs(X, y, horizons):
    """Rows of [EIS at n | cycles ahead] -> SOH at n+h, built within each cell."""
    feats, targets, hs = [], [], []
    for cell in pd.unique(X.index.get_level_values("channel")):
        Xc = X.xs(cell, level="channel").sort_index()
        yc = y.xs(cell, level="channel").sort_index().to_numpy()
        cyc = Xc.index.to_numpy(dtype=float)
        A = Xc.to_numpy(dtype=float)
        for h in horizons:
            if h >= len(yc):
                continue
            i = np.arange(len(yc) - h)
            feats.append(np.column_stack([A[i], cyc[i + h] - cyc[i]]))
            targets.append(yc[i + h])
            hs.append(np.full(len(i), h))
    return np.vstack(feats), np.concatenate(targets), np.concatenate(hs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="PEIS-HC-RT")
    p.add_argument("--ns", type=int, nargs="+")
    p.add_argument("--capacity-ns", type=int)
    p.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 40])
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)

    spec = get_dataset(args.dataset, eis_ns=args.ns, capacity_ns=args.capacity_ns)
    out_dir = Path(args.out) / "xgb" / spec.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    X, y = build_dataset(load_dataset(spec), spec)
    print(f"{spec.name}: X={X.shape}, horizons={args.horizons}")

    rows = []
    for cell, train, test in loso_folds(X):
        F_tr, t_tr, _ = make_pairs(X[train], y[train], args.horizons)
        F_te, t_te, h_te = make_pairs(X[test], y[test], args.horizons)
        pred = XGBRegressor(**PARAMS).fit(F_tr, t_tr).predict(F_te)
        rows.append(pd.DataFrame({"channel": cell, "horizon": h_te, "y_true": t_te, "y_pred": pred}))
        print(f"  fold {cell} done ({time.time() - t0:.0f}s)")

    preds = pd.concat(rows, ignore_index=True)
    curve = pd.DataFrame([
        {"horizon": h, "n": len(g), **metrics.scores(g["y_true"], g["y_pred"])}
        for h, g in preds.groupby("horizon")
    ])
    curve.to_csv(out_dir / "multistep_curve.csv", index=False)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    for ax, col, label in zip(axes, ("r2", "rmse"), ("pooled R2", "pooled RMSE (SOH)")):
        ax.plot(curve["horizon"], curve[col], "o-", color=plots.SERIES[0], lw=2, ms=6)
        ax.set_xlabel("Forecast horizon (EIS steps ahead)")
        ax.set_ylabel(label)
        plots.style_axes(ax)
    fig.suptitle(f"XGB multi-step forecasting, {spec.name} (Ns {list(spec.eis_ns)})")
    fig.tight_layout()
    fig.savefig(out_dir / "multistep_curve.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
