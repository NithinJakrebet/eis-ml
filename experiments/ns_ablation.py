"""EIS-state ablation: which Ns step carries the SOH signal?

Runs the LOSO experiment for Ns=1 only (discharged rest), Ns=6 only (charged
rest) and Ns=1+6 for each model, then gathers the summaries into one table
and a bar chart. Each run is a normal ``run_loso`` invocation, so every
per-run artefact is kept under ``results/ablation/ns/<model>/<dataset>_ns<steps>/``.

    python experiments/ns_ablation.py --dataset PEIS-HC-RT [--models gpr xgb]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root holds the modules
import plots  # noqa: E402
import run_loso  # noqa: E402  (sibling script)

NS_SETS = ((1,), (6,), (1, 6))


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--dataset", default="PEIS-HC-RT")
    p.add_argument("--models", nargs="+", default=["gpr", "xgb"])
    p.add_argument("--param", action="append", default=[], help="passed through to run_loso")
    p.add_argument("--out", default="results/ablation/ns")
    args = p.parse_args(argv)

    rows = []
    for model in args.models:
        for ns in NS_SETS:
            run_argv = ["--dataset", args.dataset, "--model", model, "--ns", *map(str, ns), "--out", args.out]
            for item in args.param:
                run_argv += ["--param", item]
            print(f"\n=== {model} Ns={ns} ===")
            run_loso.main(run_argv)
            tag = f"{args.dataset}_ns{'-'.join(map(str, ns))}"
            s = json.loads((Path(args.out) / model / tag / "summary.json").read_text())
            low = next((b for b in s["by_soh_band"] if b["band"].startswith("[0.0")), {})
            rows.append({
                "model": model,
                "eis_ns": "+".join(map(str, ns)),
                "n_features": s["n_features"],
                "pooled_rmse": s["pooled"]["rmse"],
                "pooled_r2": s["pooled"]["r2"],
                "mean_per_cell_rmse": s["mean_per_cell_rmse"],
                "rmse_soh_below_0.4": low.get("rmse", np.nan),
            })

    table = pd.DataFrame(rows)
    out = Path(args.out)
    table.to_csv(out / f"ns_ablation_{args.dataset}.csv", index=False)
    print("\n" + table.to_string(index=False, float_format=lambda v: f"{v:.4f}"))

    piv = table.pivot(index="eis_ns", columns="model", values="pooled_rmse").reindex(["1", "6", "1+6"])
    fig, ax = plt.subplots(figsize=(6, 4))
    x = np.arange(len(piv))
    w = 0.8 / len(piv.columns)
    for i, m in enumerate(piv.columns):
        bars = ax.bar(x + (i - (len(piv.columns) - 1) / 2) * w, piv[m], w * 0.92,
                      color=plots.SERIES[i], label=m.upper())
        ax.bar_label(bars, fmt="%.3f", fontsize=8, color=plots.MUTED, padding=2)
    ax.set_xticks(x)
    ax.set_xticklabels([f"Ns {v}" for v in piv.index])
    ax.set_ylabel("pooled RMSE (SOH)")
    ax.set_title(f"EIS-state ablation, {args.dataset} (LOSO)", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    plots.style_axes(ax)
    fig.tight_layout()
    fig.savefig(out / f"ns_ablation_{args.dataset}.png", dpi=150, bbox_inches="tight")
    print(f"Saved -> {out}")


if __name__ == "__main__":
    main()
