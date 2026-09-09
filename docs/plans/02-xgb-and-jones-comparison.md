# Plan 02 — XGBoost evaluation + Jones (2022) comparison

## Context
We have a Zhang-faithful GPR pipeline; now bring XGBoost in as the second model,
benchmarked against Jones et al. (Nat. Commun. 2022, "Impedance-based forecasting").
Key constraint: **our data uses a constant charge/discharge protocol**, so we cannot
use Jones' "action" (future protocol) predictor, and cannot reproduce their central
"SOH-is-insufficient under variable usage" thesis. Our task is SOH estimation
(current capacity from current EIS) — the regime where EIS-only is strong.

## Avoid redundancy (already exists)
- `scripts/algorithms/xgb.py`: `train_ensemble_model` + `predict_ensemble` (ensemble
  mean/std = Jones' uncertainty method). REUSE; do not rewrite.
- `notebooks/experiments/xgb/{xgb_loso,xgb_charged-state}.ipynb`: prior 14-cell LOSO +
  Ns6-only runs (params already match Jones: 500 est / depth 100 / lr 0.1).
- `results/xgb/xgb_8_fold_cv_results.csv`: per-cell results (mean R2=0.827, B5 negative).
A plain XGB LOSO is DONE — the value-add below is unified comparison + paper-inspired extensions.

## Methods vs Jones (for the writeup)
- Match: ensemble XGB, 500 est / depth 100, std-based uncertainty.
- Differ: our state is dual-SoC 140-dim (Ns1+Ns6) vs Jones' single-state 114-dim;
  our target is current SOH vs Jones' future Qn+j; CV is LOSO vs leave-2-out.
- Implication: our strong EIS-only scores are consistent with Jones once constant
  protocol is accounted for (their EIS-only R2=0.05 was driven by protocol variance).

## Proposed work (pick scope via the question I will ask)
1. **Unified GPR vs XGB comparison** (cheap, high value): run XGB LOSO reusing
   `xgb.py`, persist predictions+std to `results/xgb/outputs/`, report per-cell +
   **pooled R2** + the same calibration/SOH-band diagnostics as GPR. One comparison table.
2. **Multi-step forecasting, horizon-as-action** (flagship new, adapts Jones Fig 4):
   target SOH_{n+j} = f(EIS_n, j); sweep j, plot R2 vs horizon. Requires a
   feature-builder that pairs EIS at cycle n with capacity at cycle n+j within a cell.
3. **Ensemble confidence-vs-error curve** (Jones Fig 6c) + **data efficiency**
   (Fig 5: error vs #training cells) using the XGB ensemble std.
4. **State-representation ablation** (Jones Table 1 style): EIS-only vs scalar-SOH
   baseline; Ns1 vs Ns6 vs dual.

## Workflow
- Notebook-first in `notebooks/experiments/xgb/` (gitignored); promote successful
  runs to `experiments/xgb/*.py`. Reuse `build_model_input` + `scripts/plots.py`.
- Keep monitoring the in-flight GPR diagnostics re-run; do XGB after it lands so the
  comparison uses the new GPR predictions.

## Verification
- `results/xgb/outputs/` has predictions + per-cell CSVs and calibration PNGs.
- A single GPR-vs-XGB table (pooled R2, RMSE, mean per-cell) on PEIS-HC-RT.
- Multi-step: R2-vs-horizon curve saved; sane monotonic-ish decay.
