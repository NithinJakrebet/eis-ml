"""Multi-step SOH forecasting with the horizon as the "action" (Jones et al. Fig. 4).

Our cells all follow one constant protocol, so the future charge/discharge
programme cannot be an input the way it is in Jones et al. The analogue is
the forecast horizon: from the EIS sweep at cycle n plus the number of cycles
ahead, predict SOH at cycle n+h.

Two targets are supported. ``level`` predicts SOH(n+h) directly from the
spectrum. ``delta`` predicts the change SOH(n+h) - SOH(n); the level is
recovered as SOH(n) + delta, so it assumes today's SOH is known. ``--anchor``
gives the level model the same information (SOH(n) as a feature) for a fair
comparison. Because SOH moves slowly, a high level R2 can hide a model
that simply repeats today's SOH, so both targets are scored against the
**persistence baseline** SOH(n+h) = SOH(n): ``skill`` is the fractional RMSE
reduction over persistence, and ``r2_delta`` measures whether the *change*
is predicted at all.

    python experiments/xgb_multistep.py --dataset PEIS-HC-RT --target delta [--ns 6] [--horizons 1 5 10 20 40]
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root holds the modules
import metrics  # noqa: E402
import plots  # noqa: E402
from datasets import get_dataset, load_dataset  # noqa: E402
from features import build_dataset  # noqa: E402
from splits import loso_folds  # noqa: E402

warnings.filterwarnings("ignore")

PARAMS = {"n_estimators": 400, "max_depth": 6, "learning_rate": 0.1, "subsample": 0.8,
          "colsample_bytree": 0.8, "tree_method": "hist", "n_jobs": 4, "random_state": 42}


def make_pairs(X, y, horizons):
    """Rows of [EIS at n | cycles ahead] with SOH at n, SOH at n+h and the horizon, per cell."""
    feats, now, future, hs = [], [], [], []
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
            now.append(yc[i])
            future.append(yc[i + h])
            hs.append(np.full(len(i), h))
    return np.vstack(feats), np.concatenate(now), np.concatenate(future), np.concatenate(hs)


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="PEIS-HC-RT")
    p.add_argument("--ns", type=int, nargs="+")
    p.add_argument("--capacity-ns", type=int)
    p.add_argument("--target", choices=["level", "delta"], default="level")
    p.add_argument("--anchor", action="store_true",
                   help="also feed today's SOH(n) as a feature (the delta target uses it implicitly)")
    p.add_argument("--horizons", type=int, nargs="+", default=[0, 1, 2, 5, 10, 20, 40])
    p.add_argument("--out", default="results")
    args = p.parse_args(argv)
    if args.target == "delta":
        args.horizons = [h for h in args.horizons if h > 0]  # delta at h=0 is identically zero

    spec = get_dataset(args.dataset, eis_ns=args.ns, capacity_ns=args.capacity_ns)
    out_dir = Path(args.out) / "xgb" / spec.tag
    out_dir.mkdir(parents=True, exist_ok=True)
    stem = "multistep_curve" if args.target == "level" else "multistep_delta_curve"
    if args.anchor:
        stem += "_anchored"

    t0 = time.time()
    X, y = build_dataset(load_dataset(spec), spec)
    print(f"{spec.name}: X={X.shape}, target={args.target}, anchor={args.anchor}, horizons={args.horizons}")

    rows = []
    for cell, train, test in loso_folds(X):
        F_tr, now_tr, fut_tr, _ = make_pairs(X[train], y[train], args.horizons)
        F_te, now_te, fut_te, h_te = make_pairs(X[test], y[test], args.horizons)
        t_tr = fut_tr - now_tr if args.target == "delta" else fut_tr
        if args.anchor:
            F_tr, F_te = np.column_stack([F_tr, now_tr]), np.column_stack([F_te, now_te])
        pred = XGBRegressor(**PARAMS).fit(F_tr, t_tr).predict(F_te)
        level = now_te + pred if args.target == "delta" else pred
        rows.append(pd.DataFrame({
            "channel": cell, "horizon": h_te, "soh_now": now_te, "y_true": fut_te,
            "y_pred": level, "delta_true": fut_te - now_te, "delta_pred": level - now_te,
        }))
        print(f"  fold {cell} done ({time.time() - t0:.0f}s)")

    preds = pd.concat(rows, ignore_index=True)
    preds.to_csv(out_dir / f"{stem}_predictions.csv", index=False)

    curve = []
    for h, g in preds.groupby("horizon"):
        level = metrics.scores(g["y_true"], g["y_pred"])
        persist = metrics.scores(g["y_true"], g["soh_now"])
        delta = metrics.scores(g["delta_true"], g["delta_pred"])
        curve.append({
            "horizon": h, "n": len(g),
            "rmse": level["rmse"], "r2": level["r2"],
            "rmse_persistence": persist["rmse"],
            "skill_vs_persistence": 1 - level["rmse"] / persist["rmse"] if persist["rmse"] > 0 else np.nan,
            "rmse_delta": delta["rmse"], "r2_delta": delta["r2"],
        })
    curve = pd.DataFrame(curve)
    curve.to_csv(out_dir / f"{stem}.csv", index=False)
    print(curve.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2))
    a, b, c = axes
    a.plot(curve["horizon"], curve["rmse"], "o-", color=plots.SERIES[0], lw=2, ms=6,
           label=f"XGB ({args.target}{', +SOH(n)' if args.anchor else ''})")
    a.plot(curve["horizon"], curve["rmse_persistence"], "s--", color=plots.MUTED, lw=1.5, ms=5,
           label="persistence SOH(n)")
    a.set_ylabel("RMSE on SOH level")
    a.legend(frameon=False, fontsize=8)
    b.plot(curve["horizon"], curve["skill_vs_persistence"], "o-", color=plots.SERIES[1], lw=2, ms=6)
    b.axhline(0, color=plots.MUTED, lw=0.8)
    b.set_ylabel("skill vs persistence (1 - RMSE ratio)")
    c.plot(curve["horizon"], curve["r2_delta"], "o-", color=plots.SERIES[2], lw=2, ms=6)
    c.axhline(0, color=plots.MUTED, lw=0.8)
    c.set_ylabel("R² on ΔSOH")
    for ax in axes:
        ax.set_xlabel("Forecast horizon (EIS steps ahead)")
        plots.style_axes(ax)
    fig.suptitle(f"XGB multi-step forecasting ({args.target} target), {spec.name} (Ns {list(spec.eis_ns)})",
                 fontsize=10)
    fig.tight_layout()
    fig.savefig(out_dir / f"{stem}.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {out_dir}")


if __name__ == "__main__":
    main()
