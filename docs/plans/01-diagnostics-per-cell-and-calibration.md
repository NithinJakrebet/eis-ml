# Plan 01 — Diagnostics: per-cell / per-SOH-band error + uncertainty calibration

## Context
The LOSO run gives pooled R²=0.792 / RMSE=0.146, with the parity plot showing
systematic over-prediction below ~0.5 SOH. Before changing the model we need to
know whether those low-SOH points are **wrong** or merely **honestly uncertain**
(large σ), and which cells / SOH regions carry the error.

**Redundancy check (per review):** `scripts/plots.py` already provides most of
the per-cell diagnostics, and they should be REUSED rather than rebuilt:
- `model_predictions(y_true, y_pred, y_std)` — pred-vs-true scatter + 95% CI band + residuals + mean σ.
- `capacity_vs_cycle(...)` — per-cell actual-vs-predicted trajectory with 95% CI.
- `residual_analysis_plots(...)`, `multicell_loso_predictions(cell_metrics)` — combined LOSO scatter colored per cell.
- `ard_summary(...)` — ARD weights vs frequency.

Existing per-cell CSVs (`results/gpr/gpr_8_fold_cv_results.csv`) and the
`results/plots/gpr_cross_loso/` PNGs are from the **OLD ARD/two-stage model**,
not the new Zhang-faithful isotropic model — so they don't describe the current
model and shouldn't be conflated with it.

So the genuinely NEW work is small: (a) persist the new model's predictions+σ,
(b) a *quantitative* calibration-coverage metric (plots.py only draws a CI band,
it doesn't measure coverage), and (c) RMSE-by-SOH-band.

## Steps

1. **Persist predictions** in `experiments/gpr/gpr_loso.py` + notebook: write
   `results/gpr/outputs/gpr_loso_predictions_PEIS-HC-RT.csv`
   (`cell, y_true, y_pred, y_std`) and `gpr_loso_per_cell_PEIS-HC-RT.csv`
   (per-cell rmse/mae/r2). This makes every later diagnostic a cheap re-read.

2. **(Runtime enabler) Two-stage isotropic fit** in
   `gpr.train_capacity_gpr(..., subset_size=...)`: optimize the 3 isotropic
   hyperparameters on a stratified subset, then freeze + condition on the full
   training set (`optimizer=None`). Cuts a fold from ~3 min to seconds. Default
   stays `subset_size=None` (full optimization). Validate: pooled R² within
   ~0.01 of the full-data run on a spot check.

3. **Re-run LOSO once** to regenerate predictions + σ for the new model.

4. **Diagnostics — reuse `scripts/plots.py` + add the 3 new pieces:**
   - REUSE `multicell_loso_predictions` (combined parity, per-cell R²) and
     `model_predictions` (CI band + residuals); optionally `capacity_vs_cycle`
     per held-out cell.
   - NEW: **RMSE-by-SOH-band** bar chart (e.g. [0,0.4),[0.4,0.7),[0.7,1.0]).
   - NEW: **calibration coverage** — empirical % of points inside ±1σ/±2σ vs
     nominal 68%/95%, plus a reliability curve; residual-vs-σ scatter colored by SOH.
   Save new PNGs to `results/gpr/outputs/`.

## Verification
- `results/gpr/outputs/` has predictions CSV, per-cell CSV, band + calibration PNGs.
- Two-stage vs full-data pooled R² differ < ~0.01 on spot check.
- We can state plainly whether low-SOH errors fall inside the ±2σ band.

## Out of scope (later plans)
Linear/non-zero mean function, reduced feature set, warped GP, temporal split,
and running the injectable pipeline on other Ns datasets (GEIS / sparseEIS).