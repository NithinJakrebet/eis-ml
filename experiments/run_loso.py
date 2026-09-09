"""Leave-one-cell-out SOH estimation for any dataset x model pair.

Examples::

    python experiments/run_loso.py --dataset PEIS-HC-RT --model xgb
    python experiments/run_loso.py --dataset PEIS-HC-RT --model gpr --ns 6 --ard
    python experiments/run_loso.py --dataset PEIS-HC-RT-sparseEIS --model gpr
    python experiments/run_loso.py --dataset GEIS-HC-RT --model xgb --param n_models=5

Outputs go to ``results/<model>/<dataset>_ns<steps>/``:
predictions.csv, per_cell.csv, summary.json, parity.png, calibration.png,
trajectories.png, plus ard_weights.csv/.png (GPR with --ard) or feature_importance.csv (XGB).
"""

from __future__ import annotations

import argparse
import ast
import json
import sys
import time
import warnings
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run without `pip install -e .`
from eis_ml import build_dataset, get_dataset, load_dataset, loso_folds  # noqa: E402
from eis_ml import metrics, plots  # noqa: E402
from eis_ml.features import feature_table  # noqa: E402
from eis_ml.models import MODELS, gpr, xgb  # noqa: E402

warnings.filterwarnings("ignore")


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", required=True, help="folder name under data/, e.g. PEIS-HC-RT")
    p.add_argument("--model", required=True, choices=sorted(MODELS))
    p.add_argument("--ns", type=int, nargs="+", help="EIS Ns step(s) to use (overrides the preset)")
    p.add_argument("--capacity-ns", type=int, help="Ns step holding the discharge capacity label")
    p.add_argument("--freq-range", type=float, nargs=2, metavar=("LO", "HI"))
    p.add_argument("--cells", nargs="+", help="restrict the dataset to these cells")
    p.add_argument("--param", action="append", default=[], metavar="KEY=VALUE",
                   help="model fit parameter, e.g. --param subset_size=None --param n_models=5")
    p.add_argument("--ard", action="store_true", help="GPR only: also fit the ARD diagnostic per fold")
    p.add_argument("--out", default="results", help="root output directory")
    return p.parse_args(argv)


def parse_params(items):
    out = {}
    for item in items:
        key, _, raw = item.partition("=")
        try:
            out[key] = ast.literal_eval(raw)
        except (ValueError, SyntaxError):
            out[key] = raw
    return out


def main(argv=None):
    args = parse_args(argv)
    spec = get_dataset(args.dataset, eis_ns=args.ns, capacity_ns=args.capacity_ns,
                       freq_range=args.freq_range, cells=args.cells)
    model = MODELS[args.model]
    fit_params = parse_params(args.param)
    out_dir = Path(args.out) / args.model / spec.tag
    out_dir.mkdir(parents=True, exist_ok=True)

    t0 = time.time()
    df = load_dataset(spec)
    X, y = build_dataset(df, spec)
    cells = list(pd.unique(X.index.get_level_values("channel")))
    print(f"{spec.name}: EIS Ns={list(spec.eis_ns)} capacity Ns={spec.capacity_ns} "
          f"-> X={X.shape} from {len(cells)} cells ({time.time() - t0:.1f}s)")
    print(f"model={args.model} params={fit_params or 'defaults'}  ->  {out_dir}")

    preds, ard_rows, importances = [], [], []
    for cell, train, test in loso_folds(X):
        t1 = time.time()
        bundle = model.fit(X[train], y[train], **fit_params)
        mean, std = model.predict(bundle, X[test])
        fold = pd.DataFrame({
            "channel": cell,
            "cycle": X.index[test].get_level_values("cycle"),
            "y_true": y[test].to_numpy(),
            "y_pred": mean,
            "y_std": std,
        })
        preds.append(fold)
        s = metrics.scores(fold["y_true"], fold["y_pred"])
        print(f"  [{cell:>3}] n={test.sum():4d}  RMSE={s['rmse']:.4f}  MAE={s['mae']:.4f}  "
              f"R2={s['r2']:7.3f}  ({time.time() - t1:.1f}s)")

        if args.model == "gpr" and args.ard:
            ard_rows.append(gpr.ard_weights(gpr.fit_ard(X[train], y[train])))
        if args.model == "xgb":
            importances.append(xgb.feature_importance(bundle))

    preds = pd.concat(preds, ignore_index=True)
    preds.to_csv(out_dir / "predictions.csv", index=False)

    per_cell = metrics.per_cell(preds)
    per_cell.to_csv(out_dir / "per_cell.csv", index=False)
    pooled = metrics.pooled(preds)
    bands = metrics.by_soh_band(preds)
    cov = metrics.coverage(preds)

    summary = {
        "dataset": spec.name, "eis_ns": list(spec.eis_ns), "capacity_ns": spec.capacity_ns,
        "freq_range": list(spec.freq_range), "model": args.model, "fit_params": fit_params,
        "n_features": int(X.shape[1]), "n_samples": int(len(X)), "cells": cells,
        "pooled": pooled, "mean_per_cell_rmse": float(per_cell["rmse"].mean()),
        "by_soh_band": bands.to_dict(orient="records"), "coverage": cov,
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2))

    title = f"{args.model.upper()} LOSO on {spec.name} (Ns {list(spec.eis_ns)})"
    plots.parity(preds, title=title).savefig(out_dir / "parity.png", dpi=150, bbox_inches="tight")
    plots.calibration(preds, title=title).savefig(out_dir / "calibration.png", dpi=150, bbox_inches="tight")
    plots.capacity_vs_cycle(preds).savefig(out_dir / "trajectories.png", dpi=120, bbox_inches="tight")

    if ard_rows:
        W = np.vstack(ard_rows)
        weights = feature_table(X)
        weights["w_mean"], weights["w_std"] = W.mean(axis=0), W.std(axis=0)
        weights.sort_values("w_mean", ascending=False).to_csv(out_dir / "ard_weights.csv", index=False)
        plots.ard_weights(weights, title=f"ARD relevance, {spec.name}").savefig(
            out_dir / "ard_weights.png", dpi=150, bbox_inches="tight")
        print("\nTop ARD features:")
        print(weights.sort_values("w_mean", ascending=False).head(10).to_string(index=False))
    if importances:
        imp = feature_table(X)
        imp["importance"] = np.mean(importances, axis=0)
        imp.sort_values("importance", ascending=False).to_csv(out_dir / "feature_importance.csv", index=False)

    print("\nPer-cell (sorted by RMSE):")
    print(per_cell.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"\nPooled: RMSE={pooled['rmse']:.4f}  MAE={pooled['mae']:.4f}  R2={pooled['r2']:.3f}  "
          f"(n={pooled['n']})   mean per-cell RMSE={summary['mean_per_cell_rmse']:.4f}")
    print(bands.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print(f"Coverage: {cov['within_1sigma']:.1%} within 1 sigma, {cov['within_2sigma']:.1%} within 2 sigma "
          f"(nominal 68% / 95%)")
    print(f"Saved -> {out_dir}  ({time.time() - t0:.0f}s total)")


if __name__ == "__main__":
    main()
