# Frequency-band ablation: GPR/ARD vs XGB importance disagreement (2026-09-10)

**Branch:** `frequency-band-ablation`. **Code:** `features.select_freq`, `experiments/band_ablation.py`.
Results: `results/ablation/bands/band_ablation_PEIS-HC-RT.csv/.png`.

## The question
ARD (GPR diagnostic) put 50% of relevance on 1-100 Hz imaginary impedance and ~1% below 1 Hz;
XGB gain importance put 94% on Re(Z) above 2 kHz. Both are model-specific measures, so the
test here is model-agnostic: LOSO with features restricted to one band ("only") or with one
band removed ("drop"). Bands: <1 Hz (diffusion), 1-100 Hz (charge transfer), 100 Hz-2 kHz
(SEI / film), >2 kHz (ohmic / contact). GPR: isotropic, 2 restarts; XGB: 5-member ensemble.

## Pooled LOSO RMSE (SOH), PEIS-HC-RT, Ns 1+6

| features | n | GPR | XGB |
|---|---|---|---|
| all | 140 | 0.132 | 0.042 |
| only <1 Hz | 12 | **0.048** | 0.069 |
| only 1-100 Hz | 62 | 0.196 | 0.050 |
| only 100 Hz-2 kHz | 42 | 0.112 | 0.052 |
| only >2 kHz | 24 | 0.140 | 0.051 |
| drop <1 Hz | 128 | 0.127 | 0.043 |
| drop 1-100 Hz | 78 | 0.122 | 0.042 |
| drop 100 Hz-2 kHz | 98 | 0.128 | 0.041 |
| drop >2 kHz | 116 | 0.172 | 0.043 |

## Reading
- **XGB: the SOH signal is redundant across the spectrum.** Every single band alone gives
  R² > 0.95 and removing any one band changes RMSE by < 0.002. Impedance grows everywhere as
  the cell ages, so the trees can use whichever features split most cleanly; the 94% gain on
  >2 kHz Re(Z) (ohmic / contact resistance) is a *preference*, not a dependency. Tree
  importance therefore says little about electrochemistry here.
- **GPR: accuracy depends on dimensionality, not on the band.** The 12 sub-1 Hz features alone
  beat all 140 by almost 3x (0.048 vs 0.132), even though ARD assigned that band ~1% relevance.
  A single isotropic length scale over 140 standardised features cannot weight them, and
  ARD fit on 300 rows with 140 length scales is too noisy to trust per band. This is the same
  failure the `gpr-mean-function` branch fixes from the other side (linear mean: 0.050).
- The one band GPR really needs among the others is >2 kHz (dropping it: 0.132 -> 0.172),
  consistent with ohmic resistance being the cleanest monotone aging marker; the 1-100 Hz
  band alone is the *worst* for GPR (0.196) despite ARD's ranking.
- Resolution: the two importance measures were answering different questions (which features
  the trees split on vs. which length scales the GP shortened). Neither identifies a uniquely
  informative band; for physics claims use ablations like this one, not importances. For GPR
  specifically, use few features (a low-frequency handful, or ARD top-k inside the fold) or a
  mean function; for XGB any band is fine and the charged-state sweep alone suffices.
