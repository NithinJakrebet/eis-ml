"""Conformal calibration of an existing LOSO run, without retraining.

Reads ``predictions.csv`` from one or more result folders, adds the
leave-one-cell-out conformal sigma (see ``metrics.conformal_scale``), rewrites
the predictions and ``summary.json`` and draws ``calibration_conformal.png``.

    python experiments/calibrate.py results/gpr/PEIS-HC-RT_ns1-6 results/xgb/PEIS-HC-RT_ns1-6
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # repo root holds the modules
import metrics  # noqa: E402
import plots  # noqa: E402


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("folders", nargs="+", help="result folders containing predictions.csv")
    p.add_argument("--alpha", type=float, default=0.05, help="1 - target coverage")
    args = p.parse_args(argv)

    rows = []
    for folder in map(Path, args.folders):
        preds = pd.read_csv(folder / "predictions.csv")
        preds = metrics.calibrate(preds.drop(columns=["sigma_scale", "y_std_cal"], errors="ignore"), args.alpha)
        preds.to_csv(folder / "predictions.csv", index=False)

        raw = metrics.coverage(preds)
        conf = metrics.conformal_summary(preds, args.alpha)
        summary_path = folder / "summary.json"
        summary = json.loads(summary_path.read_text()) if summary_path.exists() else {}
        summary["conformal"] = conf
        summary_path.write_text(json.dumps(summary, indent=2))

        title = f"{summary.get('model', folder.parent.name).upper()} LOSO on {summary.get('dataset', folder.name)}"
        plots.calibration(preds, title=title + ", conformal sigma", std_col="y_std_cal").savefig(
            folder / "calibration_conformal.png", dpi=150, bbox_inches="tight")
        rows.append({
            "run": str(folder), "raw_within_2sigma": raw["within_2sigma"],
            "conformal_coverage": conf["coverage"], "sigma_scale": conf["median_sigma_scale"],
            "mean_half_width": conf["mean_half_width"], "worst_cell_coverage": conf["per_cell_coverage_min"],
        })
    print(pd.DataFrame(rows).to_string(index=False, float_format=lambda v: f"{v:.3f}"))


if __name__ == "__main__":
    main()
