# EIS-ML Project Context

Orientation document for Claude Code (and the `battery-ml` agent). Read it before
any non-trivial task in this repo. The README covers setup and commands; this file
covers *why* things are the way they are.

**Repo:** https://github.com/NithinJakrebet/eis-ml

---

## Goal

Predict lithium-ion battery **state of health (SOH)** from a non-invasive
**electrochemical impedance spectroscopy (EIS)** sweep, instead of from a full
charge/discharge cycling test. Each spectrum is a fast electrical snapshot of the
cell; ML learns spectrum -> capacity. Longer term: multi-step forecasting of SOH
(`experiments/xgb_multistep.py` is the first version) and, if data with varying
protocols arrives, Jones-style forecasting with the future protocol as an input.

---

## Methodological anchors

### Zhang et al., Nature Communications 11:1706 (2020)
- 12 Eunicell LR2032 coin cells, EIS at 9 states along the CC-CV profile; state V
  (rest after full charge) is most predictive.
- 120 features = 60 frequencies x {Re(Z), -Im(Z)}; inputs z-scored with training stats.
- **Capacity predictor:** GPR with an *isotropic* squared-exponential kernel and a
  learned Gaussian noise term (GPML `covSEiso` + `likGauss`, zero mean).
- **ARD** (`covSEard`, one length scale per feature) is a *separate diagnostic* used
  only to rank frequencies; relevance = `exp(-length_scale)`. It found 17.80 Hz and
  2.16 Hz to be the salient frequencies (charge-transfer regime).
- The MATLAB reference code lives outside this repo in `../Zhang/Code-Matlab/`
  (`Multi_T_EIS_Capacity_GPR.m`, `ARD_GPR.m`); `algorithms/gpr.py` mirrors it.

### Jones et al., Nature Communications 13:4806 (2022)
- 88 coin cells cycled with *randomly varying* currents; forecasts future capacity
  from one EIS sweep plus the future protocol ("action"), using an ensemble of
  XGBoost regressors whose spread is the uncertainty.
- Their headline that "EIS alone is insufficient" is driven by protocol variance.
  **Our cells all follow one constant protocol**, so the action input has no
  variance and our task reduces to SOH estimation, where EIS alone is strong.
  `algorithms/xgb.py` mirrors the ensemble; the forecast horizon plays the
  role of the action in `experiments/xgb_multistep.py`.

Other background papers in `../Research Papers/`: Messing 2021 (relaxation + EIS),
Gasper 2022 (featurisation survey), Ling 2022 (battery informatics review),
van Vlijmen 2023 (interpretable aging modes), and a GPR tutorial.

---

## Data

Raw CSVs live in `data/` (gitignored; download link in the README) or wherever
`EIS_ML_DATA` points. One CSV per cell, Bio-Logic long form. Nothing in the
package cares where the CSVs come from beyond `datasets.load_cell`, so a
Dropbox or Drive sync only needs to land files in that layout.

| dataset | cells | EIS steps (`eis_ns`) | capacity step (`capacity_ns`) | notes |
|---|---|---|---|---|
| PEIS-HC-RT | A1-A8, B1-B6 | 1 (discharged), 6 (charged) | 8 | primary; ~2,785 cycles, 140 features |
| GEIS-HC-RT | B7, B8 | 1, 6 | 8 | galvanostatic, only ~55 cycles per cell, SOH barely moves |
| PEIS-HC-RT-sparseEIS | A1-A8 | 5 | 3 | EIS every ~10 cycles (15-16 per cell); **A4 ran 790 cycles and died** (capacity -> 0 after ~cycle 175 while EIS kept being measured, 71 sweeps), which dominates its error |
| Na_NoEIS | A1, A2 | none | n/a | sodium cells, no impedance columns |

These are encoded as `DatasetSpec` presets in `datasets.py`. **The EIS step
is injectable**: `get_dataset("PEIS-HC-RT", eis_ns=[6])` or `--ns 6` on the CLI.
A new folder needs no code, only `--ns` and `--capacity-ns` (or a new preset).

Data quirks handled by the loader / feature builder:
- `-Im(Z)/Ohm` is exported as `#NAME?` (spreadsheet mangling of the minus sign).
- One cycle per PEIS cell carries the Ns=1 sweep twice; duplicates collapse to the median.
- Ns=1 sweeps go down to 0.009 Hz; the default `freq_range=(0.2, 20000)` keeps 37 of 48.
- Cycle 0 (formation) and the final (truncated) cycle are dropped.
- The last cycle of some cells lacks the capacity step; `build_dataset` inner-joins.

---

## Pipeline (modules at the repo root)

```
load_dataset(spec) -> df           long form, all cells, 'channel' column
eis_features(df, eis_ns)           pivot once for ALL cells -> X indexed (channel, cycle), columns (ns, part, freq)
capacity_labels(df, capacity_ns)   max Capacity/mA.h per (channel, cycle) -> SOH = cap / first cycle of that cell
build_dataset(df, spec) -> X, y    inner join of the two
loso_folds(X)                      (cell, train_mask, test_mask) per cell
MODELS[name].fit / .predict        gpr or xgb; predict returns (mean, std)
metrics.* / plots.*                pooled + per-cell scores, bands, coverage; figures
```

Invariants (enforced by code and `tests/test_features.py`):
1. One row per (channel, cycle); channels are never mixed.
2. A cycle is kept only when every requested Ns step was measured in it.
3. Column order: per requested Ns, all `re` then all `im`, frequency ascending.
   `X.columns` **is** the layout; never hardcode 37/33 offsets. Use
   `feature_table(X)` to join per-feature results back to frequencies.
4. Features are built once for the whole dataset, then split by cell. This is
   ~150x faster than the old per-fold Python loops and guarantees train and test
   share the same column grid. It is leakage-free because feature construction is
   purely per-row; standardisation happens inside `algorithms.gpr.fit` on training rows.
5. Capacity (the target), energy and SOC never enter `X`.

### Historical bugs, do not regress
- **Channel mixing:** an early loader concatenated cells and did `.iloc[0]` on an
  unfiltered frame. Every per-cycle operation now runs on a (channel, cycle) index.
- **Action-vector leakage:** a former "action vector" used capacity. Legal action
  signals, if ever reintroduced, are currents (Ns=3 charge, Ns=8 discharge), not capacity.
- **Frequency grid drift:** the old code derived the grid separately for train and
  test folds, which silently misaligned columns on GEIS. Building once fixes it.

---

## Models (`algorithms/`)

- **algorithms.gpr.fit** - Zhang-faithful: `Constant * RBF(isotropic) + WhiteKernel(learned)`,
  `normalize_y=True`. `subset_size=500` optimises the 3 hyperparameters on a
  stratified subset, then conditions on all training rows (`optimizer=None`).
  NaN features are imputed to the column mean after standardisation.
- **gpr.fit_ard / ard_weights** - diagnostic only: per-feature length scales on a
  300-row subset; relevance `exp(-l)`. Per-frequency weights are noisy across folds;
  aggregate conclusions (which Ns, Re vs Im, which band) are robust.
- **xgb.fit** - Jones-style ensemble: `n_models=10` regressors, each on a different
  80% subset with its own seed; mean = prediction, std = uncertainty. Handles NaN.

Old two-stage ARD predictor and its top-k frequency selection were removed in the
September 2026 refactor; their outputs are under `results/legacy/`.

---

## Evaluation

- **LOSO only.** Train on N-1 cells, test on the held-out cell.
- **RMSE in SOH units is the headline.** Per-cell R2 collapses for cells with tiny
  SOH variance (B5, B6, GEIS cells), so report pooled R2 next to the per-cell table.
- `summary.json` adds RMSE by SOH band ([0,0.4), [0.4,0.7), [0.7,1]) and sigma
  coverage (fraction of residuals inside 1 and 2 predicted sigma).

Findings so far (PEIS-HC-RT, 14 cells, Ns=1+6, September 2026 runs):
- **GPR (isotropic):** pooled RMSE 0.132, R2 0.83; mean per-cell RMSE 0.10. Error
  concentrates below SOH 0.4 (RMSE 0.27, bias +0.17): under LOSO the GP reverts
  toward the training mean when a held-out cell degrades further than any training
  cell. Cells B1/B3/B4 (deep degradation, few cycles) carry most of the error.
  Only 54% of residuals fall inside 2 sigma, so the GP is over-confident there.
- **XGB ensemble:** pooled RMSE 0.042, R2 0.98; mean per-cell RMSE 0.037. Flat error
  across SOH bands. Ensemble spread badly under-covers (25% inside 2 sigma), so it is
  a ranking of uncertainty, not a calibrated one.
- **ARD (GPR diagnostic):** Ns=6 (charged) carries 73% of relevance, imaginary part
  72%, the 1-100 Hz band 50%, the < 1 Hz tail ~1%. Top features are -Im(Z) at
  31.6 / 17.8 / 23.7 Hz in the charged state; 17.80 Hz is exactly Zhang's salient
  frequency. Per-frequency ranks are noisy across folds (std ~ mean).
- **XGB gain importance** disagrees: 94% sits in Re(Z) above 2 kHz (10 kHz and
  7.45 kHz), i.e. ohmic / contact resistance. Open question whether trees exploit
  a cell-specific offset the GP's isotropic kernel cannot.
- **Multi-step XGB:** pooled R2 0.983 at h=0 decays only to 0.977 at 40 cycles ahead;
  SOH changes slowly, so test skill on delta-SOH next.
- Cells B5 and B6 barely degrade (SOH std 0.018), so their per-cell R2 is meaningless.
Current numbers: `results/<model>/PEIS-HC-RT_ns1-6/summary.json`.

---

## Open questions / next steps

1. Ns ablation with the injectable step: Ns=1 vs Ns=6 vs both, GPR and XGB
   (`--ns 1`, `--ns 6`); confirm the ARD claim that Ns=6 carries the signal.
2. GPR low-SOH bias: non-zero mean function, reduced physics-motivated feature set
   (top ARD frequencies), or a warped/heteroscedastic GP.
3. Forecasting skill on delta-SOH rather than SOH level; temporal split within cells.
4. Why is B5 hard? Why does XGB weight high frequencies (SEI/contact resistance?)
5. Extra per-cycle variables (e.g. charge current, rest voltage) as features:
   request them via `load_cell(columns=...)`, build a (channel, cycle)-indexed frame,
   and join it to `X`. Keep capacity-derived quantities out.
6. Cloud data: a fetch step that mirrors `<dataset>/<cell>.csv` into a local cache;
   set `EIS_ML_DATA` to it.
7. sparseEIS: decide how to treat dead-cell cycles (A4) before comparing protocols.

---

## Conventions

- Python env: conda `eis-ml-conda`; `pip install -r requirements.txt`; `pytest` before committing.
- Prototype in `notebooks/` (untracked, add the repo root to `sys.path`); promote to `experiments/*.py`.
- Results are small tracked artefacts under `results/<model>/<dataset>_ns<steps>/`.
- Don't commit `data/`, `notebooks/`, `models/`, `plots/`, `mlruns/`.
- Terse, function-oriented code; no class hierarchies or config frameworks. A
  `DatasetSpec` dataclass and a `MODELS` dict are the only indirection.
