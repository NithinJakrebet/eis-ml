# Multi-step forecasting: predict the change in SOH, not the level (2026-09-10)

**Branch:** `multistep-delta-soh`. **Code:** `experiments/xgb_multistep.py --target {level,delta} [--anchor]`.
Every run now scores against the **persistence baseline** SOH(n+h) = SOH(n) and reports
R² on ΔSOH, which is what "forecasting skill" means when SOH moves slowly.

## Three formulations (PEIS-HC-RT, Ns 1+6, 14-cell LOSO, one XGB per fold)

| horizon h | persistence RMSE | level, EIS only | delta (EIS, anchored on SOH(n)) | level + SOH(n) feature |
|---|---|---|---|---|
| 1 | 0.011 | 0.042 (skill -295%) | 0.011 (-6%) | 0.019 (-76%) |
| 5 | 0.033 | 0.042 (-27%) | 0.018 (+47%) | 0.024 (+27%) |
| 10 | 0.061 | 0.043 (+30%) | 0.026 (+57%) | 0.031 (+49%) |
| 20 | 0.114 | 0.049 (+57%) | 0.040 (+65%) | 0.037 (+68%) |
| 40 | 0.189 | 0.050 (+74%) | 0.056 (+70%) | 0.044 (+77%) |
| R² on ΔSOH at h=40 | – | 0.90 | 0.87 | 0.92 |

RMSE in SOH units; skill = 1 - RMSE / persistence RMSE. Files:
`results/xgb/PEIS-HC-RT_ns1-6/multistep_curve*.csv/.png` and `*_predictions.csv`.

## Reading
- The earlier headline "R² 0.98 out to 40 cycles" was the *level* and mostly reflects that SOH
  barely moves. The honest question is whether the spectrum predicts the *change*, and it does:
  **R² on ΔSOH is 0.87-0.92 at 40 cycles ahead** and the RMSE is 3-4x below persistence.
- At 1-2 cycles ahead nothing beats persistence: the per-cycle fade (~0.01 SOH) is at the noise
  floor of the label itself. Skill appears from ~5 cycles (delta target) or ~10 cycles (EIS only).
- Knowing today's SOH helps at short horizons (the delta target inherits it exactly) and the
  spectrum helps at long ones; giving the level model SOH(n) as a feature combines both and is
  the best at h >= 20. Which formulation to use depends on whether a capacity measurement is
  available at forecast time; EIS-only is the realistic deployment case.
- Next: Ns=6-only spectra (`--ns 6`), and per-cell skill (the B cells degrade fastest and
  dominate long-horizon errors).
