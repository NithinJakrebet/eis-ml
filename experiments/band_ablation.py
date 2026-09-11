"""Frequency-band ablation: where in the spectrum does each model get its signal?

GPR's ARD diagnostic points at the 1-100 Hz charge-transfer region while
XGBoost's gain importance sits almost entirely above 2 kHz (ohmic / contact
resistance). Importances are model-specific, so this script tests both models
on the same footing: LOSO with features restricted to one band ("only") or with
one band removed ("drop"), plus the full spectrum as reference.

    python experiments/band_ablation.py --dataset PEIS-HC-RT [--models gpr xgb]

Writes ``results/ablation/bands/band_ablation_<dataset>.csv`` and ``.png``.
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

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root holds the modules
import metrics  # noqa: E402
import plots  # noqa: E402
from algorithms import MODELS  # noqa: E402
from datasets import get_dataset, load_dataset  # noqa: E402
from features import build_dataset, select_freq  # noqa: E402
from splits import loso_folds  # noqa: E402

warnings.filterwarnings("ignore")

BANDS = {
    "<1Hz": (0.0, 1.0),           # solid-state diffusion (Warburg)
    "1-100Hz": (1.0, 100.0),      # charge transfer / double layer
    "100Hz-2kHz": (100.0, 2000.0),  # SEI / surface film
    ">2kHz": (2000.0, np.inf),    # ohmic, contact, electrolyte
}
# Fast settings: the ablation compares configurations, it is not the headline run.
FIT_PARAMS = {
    "gpr": {"n_restarts": 2},
    "xgb": {"n_models": 5, "n_jobs": 3},
}


def configs():
    yield "all", None, False
    for name, (lo, hi) in BANDS.items():
        yield f"only {name}", (lo, hi), False
    for name, (lo, hi) in BANDS.items():
        yield f"drop {name}", (lo, hi), True


def loso_rmse(model, X, y):
    yt, yp = [], []
    for _, train, test in loso_folds(X):
        bundle = model.fit(X[train], y[train], **FIT_PARAMS[model.__name__.split(".")[-1]])
        mean, _ = model.predict(bundle, X[test])
        yt.append(y[test].to_numpy())
        yp.append(mean)
    yt, yp = np.concatenate(yt), np.concatenate(yp)
    s = metrics.scores(yt, yp)
    low = yt < 0.4
    s["rmse_soh_below_0.4"] = float(np.sqrt(np.mean((yt[low] - yp[low]) ** 2))) if low.any() else np.nan
    return s


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="PEIS-HC-RT")
    p.add_argument("--ns", type=int, nargs="+")
    p.add_argument("--models", nargs="+", default=["gpr", "xgb"])
    p.add_argument("--out", default="results/ablation/bands")
    args = p.parse_args(argv)

    spec = get_dataset(args.dataset, eis_ns=args.ns)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    X, y = build_dataset(load_dataset(spec), spec)
    print(f"{spec.name}: X={X.shape}")

    t0 = time.time()
    rows = []
    for model_name in args.models:
        model = MODELS[model_name]
        for cfg, band, exclude in configs():
            Xc = X if band is None else select_freq(X, *band, exclude=exclude)
            if Xc.shape[1] == 0:
                continue
            s = loso_rmse(model, Xc, y)
            rows.append({"model": model_name, "config": cfg, "n_features": Xc.shape[1], **s})
            print(f"  {model_name:3s} {cfg:16s} feats={Xc.shape[1]:3d}  RMSE={s['rmse']:.4f}  "
                  f"R2={s['r2']:.3f}  low-SOH RMSE={s['rmse_soh_below_0.4']:.4f}  ({time.time() - t0:.0f}s)")

    table = pd.DataFrame(rows)
    table.to_csv(out / f"band_ablation_{spec.name}.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), sharey=True)
    for ax, kind in zip(axes, ("only", "drop")):
        sub = table[table["config"].str.startswith(kind) | (table["config"] == "all")]
        piv = sub.pivot(index="config", columns="model", values="rmse")
        order = ["all"] + [f"{kind} {b}" for b in BANDS]
        piv = piv.reindex([o for o in order if o in piv.index])
        x = np.arange(len(piv))
        w = 0.8 / len(piv.columns)
        for i, m in enumerate(piv.columns):
            bars = ax.bar(x + (i - (len(piv.columns) - 1) / 2) * w, piv[m], w * 0.92,
                          color=plots.SERIES[i], label=m.upper())
            ax.bar_label(bars, fmt="%.3f", fontsize=7, color=plots.MUTED, padding=2)
        ax.set_xticks(x)
        ax.set_xticklabels(piv.index, rotation=20, ha="right")
        ax.set_title(f"features restricted to one band" if kind == "only" else "one band removed",
                     fontsize=10, loc="left")
        ax.legend(frameon=False, fontsize=8)
        plots.style_axes(ax)
    axes[0].set_ylabel("pooled LOSO RMSE (SOH)")
    fig.suptitle(f"Frequency-band ablation, {spec.name} (Ns {list(spec.eis_ns)})", fontsize=10)
    fig.tight_layout()
    fig.savefig(out / f"band_ablation_{spec.name}.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
