# Uncertainty calibration via leave-one-cell-out conformal scaling (2026-09-10)

**Branch:** `uncertainty-calibration`. **Code:** `metrics.conformal_scale / calibrate / conformal_summary`,
`experiments/calibrate.py` (post-processes any results folder), `run_loso.py` now calibrates every run.

## Method
Both models ship a sigma that is far too small (GPR: 54% of residuals inside 2σ; XGB ensemble
spread: 25%). Split-conformal calibration fixes the *scale*: for held-out cell c, the multiplier
`s_c` is the finite-sample 95% quantile of |residual| / σ over the **other** cells' LOSO
predictions, and the interval is `y_pred ± s_c σ`. The held-out cell never calibrates itself,
so the reported coverage is honest. `y_std_cal = s σ / 1.96` is stored so the existing
coverage and reliability plots apply unchanged.

## Results (PEIS-HC-RT, Ns 1+6)

| run | raw within 2σ | conformal 95% coverage | σ scale | mean half-width (SOH) | worst cell coverage |
|---|---|---|---|---|---|
| GPR | 54% | 93.9% | x7.4 | 0.33 | 75% |
| XGB | 25% | 93.5% | x17.6 | 0.10 | 73% |
| GPR sparseEIS | 45% | 65% | x8.9 | 0.39 | 10% (dead cell A4) |
| XGB GEIS | 17% | 64% | x40.7 | 0.10 | 28% |

## Reading
- On the main dataset, one global scale factor gets both models to ~94% pooled coverage, so the
  raw sigmas are **rank-informative but mis-scaled**: GPR by ~7x, the XGB ensemble spread by ~18x.
- The price is width: the GPR 95% band is ±0.33 SOH on average, XGB ±0.10. XGB's calibrated band
  is both narrower and as well covered, consistent with its lower RMSE.
- Per-cell coverage still dips to ~75% on the deeply degraded B cells: the error there is
  heteroscedastic (grows with degradation) while a single multiplier cannot follow that.
  Next step if needed: scale as a function of predicted SOH (binned conformal) or of σ itself.
- On sparseEIS and GEIS the other cells do not represent the held-out one (dead cell A4, two
  near-identical GEIS cells), so calibration transfers poorly; those numbers are a warning, not a fix.
