# EIS-ML Project Context

This file is the orientation document for Claude Code (and the `battery-ml` agent) when working in this repo. Read it before doing any non-trivial task here.

**Repo:** https://github.com/NithinJakrebet/eis-ml

---

## Goal

Predict lithium-ion battery **State of Health (SOH)** — and eventually **State of Charge (SoC)** and future degradation trajectories — from non-invasive **Electrochemical Impedance Spectroscopy (EIS)** measurements, instead of from full charge/discharge cycling tests. Each EIS spectrum is a fast electrical "snapshot" of a cell's internal state; ML learns the mapping from impedance spectra to capacity.

Long-term direction: absolute capacity prediction → ΔQₙ (capacity fade per cycle) → multi-step degradation forecasting.

---

## Methodological Anchor: Zhang et al. (2020)

This project builds on **Zhang et al., *Nature Communications* 11:1706 (2020)** — "Identifying degradation patterns of lithium ion batteries from impedance spectroscopy using machine learning."

### What Zhang did
- 12 Eunicell LR2032 coin cells (LCO/graphite), cycled at 25/35/45°C, 1C charge / 2C discharge.
- EIS at **9 states** along the CC-CV profile (I = before charging, V = 15 min rest after full charge, IX = 15 min rest after full discharge, etc.). The most predictive state is **V**.
- Features per sample: **120** = 60 frequencies (0.02 Hz–20 kHz) × {Re(Z), Im(Z)}.
- **Two GPR models** (GPML MATLAB toolbox):
  - **EIS-Capacity GPR** — SE kernel with ARD, predicts normalized capacity.
  - **EIS-RUL GPR** — Linear kernel, predicts remaining useful life.
- **Train/test split:** 4 cells train, 4 cells test at 25°C; multi-temp variant adds 35C01 + 45C01 to training, 35C02 + 45C02 to test.
- **Key finding:** ARD identified just **two salient frequencies** sufficient to estimate capacity: **17.80 Hz** (91st feature) and **2.16 Hz** (100th feature). Both are in the low-frequency interfacial / charge-transfer regime.
- **Performance:** R² ≈ 0.88 for 25°C capacity; R² ≈ 0.81 for 35°C multi-temp; R² 0.68–0.96 for RUL.

### What we do differently
| Zhang | Our project |
|---|---|
| State V only (one state at a time) | **Both Ns=1 (discharged) and Ns=6 (charged)** concatenated into one vector (configurable via `ns_states`) |
| 60 freqs × Re/Im = **120 features** | 37 (Ns=1) + 33 (Ns=6) freqs × Re/Im = **140 features** when both states are used |
| GPML toolbox (MATLAB) | **scikit-learn** `GaussianProcessRegressor` (Python) |
| Train/test split by cell | **LOSO** (leave-one-cell-out) over 14 cells |
| Eunicell LR2032 LCO/graphite, 25/35/45°C | Different chemistry & protocol (PEIS-HC-RT) |
| Salient freqs: **17.80 Hz, 2.16 Hz** | Salient freqs (GPR): **4.8 Hz, 13.3 Hz** — same low-freq regime, different peaks |

---

## Data

- **Primary dataset:** `data/PEIS-HC-RT/` (PEIS, high C-rate, room temperature). All cells in this folder share the same Ns step configuration, which is what makes a clean pipeline possible.
- **Cells:** 14 channels — `A1–A8`, `B1–B6` (one CSV per cell).
- **Other folders** in the data tree may use GEIS instead of PEIS, or different Ns configurations — do **not** mix them into PEIS-HC-RT without rebuilding the pipeline:
  - `data/PEIS-HC-RT-sparseEIS/` — EIS every 10 cycles, Ns=5 only, capacity from Ns=3
  - `data/GEIS-HC-RT/` — galvanostatic EIS, same Ns=1/6 layout
  - `data/Na_NoEIS/` — no EIS
- File format: `.mpt` exported to `.csv`. Use `eclabfiles` (`pip install eclabfiles`) for MPT → CSV if needed.
- Data lives outside the repo (Dropbox / Google Drive) — `data/` is gitignored. Do not commit raw data.

### Ns states (PEIS-HC-RT)
Each cycle contains EIS measurements at multiple Ns steps within the charge/discharge sequence:
- **Ns = 1:** full discharge (low-SoC) — ~37 frequencies, ~0.25 Hz–10 kHz. More degradation-sensitive.
- **Ns = 6:** full charge (high-SoC) — ~33 frequencies, ~1 Hz–10 kHz.
- **Ns = 8:** constant-current discharge — used to extract the **capacity label** (final/max `Capacity/mA.h`).

Default pipeline concatenates **both** Ns=1 and Ns=6 (140 features per sample). The `ns_states` argument on `build_state_vector` / `build_model_input` lets you run Ns=1-only or Ns=6-only experiments — feature count changes accordingly.

---

## Repo layout

```
scripts/
  data_pipeline/
    load_single_channel.py   # Loads one cell's CSV, tags with `channel` column
    test_train_split.py      # `loso(...)`, `temporal_split`, `bin_and_split`
    main.py                  # `load_and_prepare_data` — convenience wrapper
  feature_engineering/
    state_vector.py          # `build_state_vector` (the 140-feature builder)
    main.py                  # `build_model_input` — adds normalized-capacity labels
  algorithms/
    gpr.py                   # `train_capacity_gpr_fast`, `predict_fast`, `ard_frequency_weights`
    xgb.py                   # `train_ensemble_model`, `predict_ensemble`
  evaluate.py                # `evaluate_model` (RMSE/R²/MSE/MAE), `save_results`
  plots.py                   # Degradation, ARD-summary, Nyquist plots

experiments/
  gpr/gpr_loso.py            # Full 14-cell LOSO GPR sweep (production reference)
  xgb/                       # XGBoost LOSO experiments

notebooks/                   # gitignored — interactive EDA + model dev
  data_exploration/          # capacity / frequency / nyquist EDA
  experiments/               # gpr_loso, gpr_loso_state_6, xgb_loso, xgb_charged-state, GPR_bin

results/                     # CSV + JSON metrics + ARD-weight summaries (tracked)
  gpr/, xgb/                 # per-model rollups + per-fold tables
  plots/                     # gitignored

docs/                        # this file lives here
.github/agents/              # custom agent specs (e.g., battery-ml.agent.md)
```

---

## State vector pipeline

See `scripts/feature_engineering/state_vector.py` (`build_state_vector`). **Invariants — do not break these:**

1. **One vector per (channel, cycle).** Never aggregate across channels. Channel identity must be preserved through every transform.
2. **Filter to the requested Ns states** (default `[1, 6]`, configurable via `ns_states`) before vectorization.
3. **Frequency range:** `0.2 Hz < f ≤ 20 kHz`.
4. **Group by `(channel, cycle number, freq/Hz)`** and aggregate impedance with **median** (handles duplicate rows).
5. **Feature layout for default `ns_states=[1,6]` (140 features, in this exact order):**
   - `[0:37]`   Re(Z) at Ns=1, sorted by frequency ascending
   - `[37:74]`  Im(Z) at Ns=1
   - `[74:107]` Re(Z) at Ns=6
   - `[107:140]` Im(Z) at Ns=6

   Code that maps ARD weights back to frequencies (e.g. in `experiments/gpr/gpr_loso.py`) relies on this exact layout — keep them in sync. If you change `ns_states`, the layout shrinks/changes and downstream weight-unpacking code must be updated.
6. **Source CSV columns:** `freq/Hz`, `Re(Z)/Ohm`, `-Im(Z)/Ohm`, `Ns`, `cycle number`, `Capacity/mA.h`, `I/mA`. The DataFrame must also carry a `channel` column added by `load_single_channel`.
7. Only keep `(channel, cycle)` pairs where **all** requested Ns states are present.

### Label construction
`build_model_input` (in `scripts/feature_engineering/main.py`) builds `y` as **normalized capacity** = `Capacity/mA.h at (channel, cycle, Ns=8)` ÷ `Capacity/mA.h at (channel, first_cycle, Ns=8)`. This is per-cell SOH as a fraction of initial capacity.

### Known bugs that have been fixed — do not regress
- **Channel-mixing bug.** The old loader concatenated all channels then `df[df['cycle number'] == cycle].iloc[0]` silently pulled impedance from whichever channel was first. Always filter by channel before frequency-level operations.
- **Action vector leakage.** The action vector previously used capacity at charge/discharge — but capacity is the target. The corrected action vector uses **median charge current** (Ns=3) and **median discharge current** (Ns=8).

---

## Models

See `scripts/algorithms/`.

### GPR (`gpr.py`)
- Kernel: `ConstantKernel × RBF (ARD) + WhiteKernel`. ARD gives per-feature length scales; `exp(-length_scale)` = feature importance.
- Two-stage fit: hyperparams learned on a stratified subset (`subset_size=300` by default), then refit on full data with frozen kernel.
- LOSO loop lives in `experiments/gpr/gpr_loso.py`.
- Defaults: `subset_size=300`, `normalize_y=True`, `n_restarts_optimizer=1`, `random_state=42`, white noise fixed at `0.001`. Constant and RBF length-scale bounds: `[0.01, 100.0]`.
- `top_k_freqs` hook supports ARD-based feature selection.
- Full 14-cell LOSO: ~80 min.

### XGBoost (`xgb.py`)
- Ensemble of `n_models` regressors (default 5), each trained on a different 80/20 split of the training fold with a different seed. Predictions averaged; std used as uncertainty.
- Uses built-in feature importance.
- Tends to do better than GPR on late-stage degradation; GPR is smoother in early/mid life.

### Frequency importance — current finding
- **GPR (ARD):** **4.8 Hz** and **13.3 Hz** dominate — low-frequency interfacial regime. Consistent with Zhang's 17.80/2.16 Hz finding (same band, different peaks).
- **XGBoost:** unexpectedly weights **high** frequencies — contradicts most prior literature. Possibly because every cycle in PEIS-HC-RT uses the same charge/discharge protocol (so the action vector has no variance and XGB has to lean elsewhere). Higher freqs may correspond to SEI cracking — open question.

Physical reference for frequency bands:
- 0.1–1 Hz: solid-state diffusion (Warburg)
- 1–10 Hz: charge transfer (dominant degradation signal)
- 100–1000 Hz: SEI surface film

---

## Evaluation

- **Primary CV: Leave-One-Cell-Out (LOSO).** Train on N−1 cells, test on the held-out cell. The only physically meaningful test of cross-cell generalization. Random k-fold over rows is invalid — it leaks the same cell into both splits.
- **Metrics:** RMSE, MAE, R². Prefer **RMSE** as the headline — capacity is normalized, so R² is misleading.
- Diagnostic-only splits (in `test_train_split.py`):
  - **Temporal:** train on early cycles (first 60%), test on later cycles of the same channel (tests forecasting).
  - **Capacity-binned (`bin_and_split`):** balanced by capacity decile but breaks temporal realism.

### Current LOSO results (GPR, full 14-cell run)
- See `results/gpr/gpr_8_fold_cv_results.csv` (legacy 8-fold over A-cells) and `results/gpr/state_6/gpr_8_fold_state_6_cv_results.csv` (Ns=6-only run, in progress).
- Earlier headline: Avg R² ≈ 0.904, Avg RMSE ≈ 0.0487 on normalized capacity (8-fold A-cells).
- **Cell B5** is a notable outlier for both models — under investigation.
- GPR sometimes predicts **negative** normalized capacity at end-of-life — open issue.

---

## Open questions / next steps

1. Why is **Cell B5** hard for both models?
2. Why does **XGBoost** prefer high frequencies, against literature consensus?
3. Move from absolute Qₙ to **ΔQₙ = Qₙ₊₁ − Qₙ**. Preliminary R² is low — expected, since per-cycle fade is small and noisy.
4. Compare Ns=1-only vs Ns=6-only vs dual-Ns (the `state_6` branch line of work). Ns=1 may be more degradation-sensitive; Ns=6 isolates the charged-state signal.
5. Feature simplification — ARD-selected top-k frequencies (`top_k_freqs` hook in `gpr.train_capacity_gpr_fast`).
6. Multi-protocol pipeline for the other data folders (GEIS, sparseEIS).

---

## Conventions

- **Branch naming** is freeform — examples in this repo: `xgb_loso`, `xgb_loso_state_6`, `Nithin-XGBoostModel`, `deep-feature-engineering`. Pick something descriptive of the experiment.
- Experiment tracking via MLflow (WIP — `mlruns/` is gitignored).
- Single conda environment (`python=3.12`, see `requirements.txt`).
- Don't commit notebooks, model artifacts, raw data, or plots (`.gitignore` enforces this for `notebooks/`, `data/`, `models/`, `plots/`, `results/plots/`).
- **Workflow:** prototype in `notebooks/experiments/*.ipynb`; when an experiment is worth preserving, export to `experiments/<model>/<name>.py`.

---

## Reference papers

- **Zhang et al. (2020)** — EIS + GPR + ARD for capacity and RUL. Methodological anchor. (See dedicated section above.)
- **Jones (XGB)** — snapshot-at-cycle prediction N cycles ahead via action vector. LOSO over 6–7 batteries.
- **Messing et al. (2021)** — relaxation effect + EIS for SoH on Samsung INR2170-50E (NMC).
- **Gasper et al. (2022)** — broad survey of feature-engineering pipelines × model architectures.
- **Rolston et al. (2022)** — leave-cell-out validation and physical consistency.
- **An Intuitive Tutorial to GPR** — GPR math background.

---

## When helping with this repo

- **Preserve channel identity** through every data transformation. If a function takes a DataFrame of impedance data, assume it must have a `channel` column and act per-channel.
- **Never** introduce target-derived features (e.g. capacity, energy) into the state vector or any action vector.
- Default validation is **LOSO**, not random k-fold.
- Use the **feature layout above** when working with ARD weights or XGBoost feature importances — index slices must stay consistent with the active `ns_states`.
- Prefer **RMSE** for capacity prediction; R² is misleading on normalized targets.
- If touching `build_state_vector`, run end-to-end and verify: `S.shape[1] == 140` for default `ns_states=[1,6]`, and exactly one row per `(channel, cycle)`.
- Stick to the protocol assumptions of `PEIS-HC-RT` unless explicitly building a new pipeline for `GEIS-HC-RT` / `PEIS-HC-RT-sparseEIS` (different Ns steps, different label step).
