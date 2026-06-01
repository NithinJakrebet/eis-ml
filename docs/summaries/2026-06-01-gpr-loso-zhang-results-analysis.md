# GPR LOSO (Zhang-faithful) — Results Analysis, ML & Physics

**Date:** 2026-06-01
**Experiment:** Leave-one-cell-out (LOSO) capacity (SOH) estimation on `PEIS-HC-RT` (14 cells, ~2,785 cycles).
**Model:** Isotropic squared-exponential GPR + learned Gaussian noise, conditioned on the full training set — a faithful port of Zhang's `Multi_T_EIS_Capacity_GPR.m` (`covSEiso` + `likGauss`). ARD (`covSEard`) is kept as a **separate diagnostic** for per-frequency relevance, mirroring Zhang's `ARD_GPR.m`.
**Artifacts:** `results/gpr/outputs/gpr_loso_parity_PEIS-HC-RT.png`, `results/gpr/outputs/gpr_loso_ard_weights_PEIS-HC-RT.csv`.

## Headline

**Pooled R² = 0.792, RMSE = 0.146 SOH.** This is an honest pooled metric (one R² over all held-out predictions vs. the global mean), replacing the previously reported mean-of-per-cell R², which went strongly negative purely as an artifact of low-variance cells (e.g. B5/B6 with SOH std ~0.018 blow up the per-cell R² denominator).

## ML perspective (parity plot)

Two regimes:
- **High SOH (0.7–1.0): excellent.** Tight cluster on the diagonal; this is where most cycles live and where all 14 cells overlap.
- **Low SOH (0.0–0.5): systematic over-prediction.** Predictions fan upward off the diagonal — true SOH ~0.05–0.4 is predicted ~0.4–0.7. The entire RMSE budget lives here.

The upward fan is **GP mean-reversion under extrapolation**. Three compounding causes:
1. **Coverage** — deeply-degraded states are rare and cell-specific; LOSO removes the one cell that would anchor end-of-life, so those EIS sit outside training support and the zero-mean posterior shrinks toward the training mean (~0.68).
2. **Non-stationarity** — a single global length scale can't fit both the slowly-varying healthy regime and the rapidly-changing, ill-conditioned end-of-life regime.
3. **Likely many-to-one mapping** — different degradation modes (LLI vs LAM) can yield similar EIS at different true capacities, making the EIS→SOH map non-injective at low SOH.

The arc-shaped "branches" are individual held-out cells: predictions track the trajectory shape but with a bias that grows as degradation deepens.

## Physics perspective (ARD relevance weights)

Aggregated relevance (weight = `exp(-length_scale)`, averaged over folds):

| Split | Finding |
|---|---|
| State of charge | **Ns=6 (charged) = 74%** vs Ns=1 (discharged) 26% |
| Re vs Im | **Imaginary = 71%** vs Real 29% |
| Frequency band | Mid 1–100 Hz: **49%**; HF 100 Hz–2 kHz: 28%; VHF >2 kHz: 21%; LF <1 Hz: **~1%** |
| Sparsity | Top-10 of 70 features = **50%** of total weight |

Interpretation:
- **Charged-state, mid-frequency imaginary impedance is the workhorse.** The 1–100 Hz imaginary region is the charge-transfer / double-layer arc; its growth tracks interfacial aging (SEI growth, charge-transfer resistance, LLI). Dominance in the *charged* state, where these features are most pronounced, is electrochemically sensible.
- **High-frequency (5–10 kHz) Re+Im also matter** → ohmic/contact + electrolyte/SEI series resistance, a classic aging marker.
- **The <1 Hz Warburg/diffusion tail is essentially discarded** (weights pinned at floor). Physically this region often carries LAM/solid-state-diffusion info, so this is either genuinely noisy here or a missed signal — relevant because the low-SOH failure above is plausibly a LAM regime the model can't see.

**Caveat:** 87% of the top-15 features have a cross-fold std ≥ their mean. The ARD diagnostic was fit on a 300-sample subset per fold, so individual frequency weights are high-variance. The *aggregate* story (Ns=6 + mid/high-freq imaginary) is robust; the precise per-frequency order is not. This echoes Zhang's own finding (Fig. 3c) that only a sparse handful of frequencies carry the signal.

## Next steps

**Diagnostics (cheap, first):**
- Persist a **per-cell RMSE table** and an **RMSE-by-SOH-band** breakdown to confirm error concentrates below ~0.5 SOH.
- **Uncertainty calibration**: plot predicted ±σ vs actual at low SOH — if the band covers truth, the model is *honestly uncertain* rather than confidently wrong, which changes how it can be used.

**Modeling:**
- **Non-zero / linear mean function** so extrapolation reverts to a degradation trend rather than a flat global mean — directly attacks the low-SOH over-prediction.
- **Reduced, physics-motivated feature set** (top ~10–20 frequencies, charged-state mid-freq Im) — likely generalizes better than 140 noisy dims and stabilizes ARD.
- **Warped/heteroscedastic GP** since SOH is bounded and EIS sensitivity saturates near EOL.

**Evaluation framing:**
- Complement LOSO with a **temporal (within-cell future-cycle) split** to separate "new cell" generalization from "future degradation" extrapolation.
