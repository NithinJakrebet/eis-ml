# EIS-ML Progress Update — Lab Meeting (2026-06-01)

**Focus today:** fixing the GPR baseline, making the feature pipeline dataset-agnostic,
and bringing XGBoost in benchmarked against Jones et al. (Nat. Commun. 2022).

## TL;DR
- Diagnosed the "negative R²" problem: it was largely a **metric artifact**, not a broken model.
- Rebuilt the GPR to faithfully match the Zhang paper (isotropic SE + learned noise).
- Made the pipeline handle **any EIS Ns-step configuration** (works across 3 datasets now).
- **XGBoost clearly beats GPR** for SOH estimation here (pooled R² ≈ 0.98 vs 0.79).
- **Multi-step forecasting works**: SOH predicted ≥40 cycles ahead with R² ≥ 0.977.

## 1. GPR baseline fixed (Zhang-faithful)
- Our old GPR used a 140-dim ARD kernel with fixed near-zero noise, fit on a 300-pt subset.
  The Zhang capacity model is actually an **isotropic** squared-exponential with **learned**
  noise, fit on all data — we now match that. (ARD is kept only as a feature-importance
  diagnostic, which is how Zhang used it.)
- The reported "negative R²" was mostly a **metric artifact**: healthy cells (e.g. B5/B6,
  SOH std ~0.02) have almost no variance, so per-cell R² explodes negative even at low error.
  We now report a **pooled R²** over all held-out predictions (honest denominator).
- **PEIS-HC-RT, 14-cell LOSO: pooled R² = 0.792, RMSE = 0.146 SOH.** Error concentrates at
  low SOH (deep degradation) due to GP mean-reversion under extrapolation.

## 2. Injectable-Ns feature pipeline
- EIS Ns-steps, frequency band, and capacity step are now configurable inputs (were hardcoded
  to Ns={1,6} + cap@Ns=8). Feature-layout metadata removes the hardcoded 37/33 frequency split.
- Validated end-to-end on **GEIS-HC-RT** (132 features) and **PEIS-sparseEIS** (Ns=5, 58 features) —
  not just PEIS. Surfaced and handled missing-frequency NaNs (sparse data) via imputation.

## 3. XGBoost vs Jones et al. (2022)
- Our ensemble-XGB setup matches Jones' method (ensemble → mean/uncertainty, ~500 trees).
- **Key conceptual point:** Jones predicts *future* capacity from EIS **+ the future
  charge/discharge protocol**; their headline is that EIS-alone fails (R²=0.05) under *variable*
  usage. **We have a constant protocol**, so there's no "action" input and our task reduces to
  **SOH estimation from EIS** — the regime where EIS-alone is strong. Our good numbers are
  consistent with the paper once you account for this.

| Model (PEIS-HC-RT, LOSO, pooled) | R² | RMSE (SOH) |
|---|---|---|
| GPR (Zhang isotropic) | 0.79 | 0.146 |
| **XGBoost (EIS → SOH)** | **0.98** | **0.042** |

XGB substantially outperforms GPR here — trees handle the low-SOH nonlinearity where the
GP reverts to the mean.

## 4. Multi-step forecasting (Jones Fig 4, adapted)
Since we can't use the future protocol as an input, we use the **forecast horizon** as the
"action": predict SOH(n+h) = f(EIS at n, cycles-ahead). XGB, 14-cell LOSO:

| Horizon (cycles ahead) | 0 | 1 | 5 | 10 | 20 | 40 |
|---|---|---|---|---|---|---|
| pooled R² | 0.983 | 0.983 | 0.983 | 0.982 | 0.977 | 0.977 |

R² barely decays with horizon — **better-maintained than Jones (R²≈0.75 at 40 cycles)** —
because a constant protocol removes the future-usage uncertainty that degrades their forecasts.
*Caveat:* SOH fades slowly, so part of this is that SOH(n+40) ≈ SOH(n); a follow-up should
test skill on the *change* in SOH, not just the level.

## Scientific takeaways
- For constant-protocol EIS data, a single EIS spectrum predicts current **and** near-future
  SOH very well (XGB pooled R² ≈ 0.98, holding ~40 cycles out).
- Degradation signal lives in **charged-state (Ns6), mid/high-frequency imaginary impedance**
  (ARD diagnostic + consistent with interfacial-aging electrochemistry).
- We **cannot** reproduce Jones' central "SOH-is-insufficient / need the action" thesis without
  variable charge/discharge rates — that would require new cycling experiments.

## In progress / next steps
- **Pipeline performance:** the state-vector builder is pure-Python and rebuilt every fold,
  making full LOSO take ~2h and timing out notebooks. Next: vectorize it (pandas pivot) →
  runs in minutes. This unblocks the GPR diagnostics re-run + unified GPR-vs-XGB table.
- **GPR diagnostics:** per-cell + SOH-band error + uncertainty calibration (is the low-SOH
  error "honestly uncertain" or "confidently wrong"?). Notebook built; pending the perf fix.
- **Reduced-feature / mean-function GPR** to attack the low-SOH mean-reversion.
- Optional: confidence-vs-error and data-efficiency curves (Jones Figs 5–6).

*Artifacts:* `results/gpr/outputs/`, `results/xgb/outputs/`; notebooks in
`notebooks/experiments/{gpr,xgb}/`; plans + summaries in `docs/`.
