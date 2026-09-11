# GPR low-SOH bias: linear mean function vs ARD feature reduction (2026-09-10)

**Branch:** `gpr-mean-function`. **Code:** `algorithms/gpr.py` (`fit(..., mean="linear")`: ridge
trend + GP on residuals), `run_loso.py --ard-top-k K [--ard-from DIR]` (leakage-free in-fold
feature selection; per-fold ARD weights are cached in `ard_fold_weights.csv`).
Table: `results/gpr/gpr_variants_PEIS-HC-RT.csv`; runs under `results/gpr/PEIS-HC-RT_ns1-6_*`.

## PEIS-HC-RT, Ns 1+6, 14-cell LOSO

| variant | pooled RMSE | R² | mean per-cell RMSE | RMSE / bias below SOH 0.4 |
|---|---|---|---|---|
| zero mean, 140 features (Zhang baseline) | 0.132 | 0.83 | 0.102 | 0.273 / +0.17 |
| **linear mean, 140 features** | **0.050** | **0.975** | **0.048** | **0.074 / +0.00** |
| zero mean, ARD top-10 (per fold) | 0.149 | 0.78 | 0.114 | 0.307 / +0.20 |
| zero mean, ARD top-20 | 0.145 | 0.79 | 0.113 | 0.303 / +0.20 |
| zero mean, ARD top-40 | 0.117 | 0.87 | 0.093 | 0.240 / +0.15 |
| linear mean, ARD top-20 | 0.052 | 0.974 | 0.051 | 0.081 / +0.00 |

Per-cell, the deeply degraded B cells drive the change: B3 0.39 -> 0.07, B4 0.33 -> 0.05,
B1 0.28 -> 0.10. Cells that were already good stay within +/-0.02.

## Reading
- **A linear mean function removes the low-SOH bias entirely.** Zhang's zero-mean GP reverts
  to the training mean when a held-out cell degrades beyond the training support; letting a
  ridge regression carry the trend and the GP only the residual keeps predictions on the
  degradation line. GPR now sits at 0.050 vs XGB's 0.042 and is competitive again.
- **ARD-selected features do not help GPR** (top-10/20 are *worse* than all 140; top-40 only
  slightly better). The in-fold ARD picks are unstable (only three features appear in >= 10 of
  14 folds: Ns 6 -Im(Z) at 31.6 and 17.8 Hz, Ns 6 Re(Z) at 7.5 kHz) and, as the
  `frequency-band-ablation` branch shows, the 12 sub-1 Hz features that ARD ranks near zero
  beat any ARD subset. Feature reduction should be band-based or physics-based, not ARD-based.
- Uncertainty is unchanged (~55% of residuals inside 2σ): the mean function fixes the bias,
  not the scale; combine with the `uncertainty-calibration` branch.
- Next: try `mean="linear"` with the sub-1 Hz band only (expected to be the best GPR), and
  a per-cell parity check that the linear trend does not over-extrapolate past SOH 0 at end of life.
