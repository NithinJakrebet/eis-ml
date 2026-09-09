# EIS-ML: Battery State of Health from Impedance Spectroscopy

Predict Li-ion battery **state of health (SOH)** from a single electrochemical
impedance spectroscopy (EIS) sweep instead of a full charge/discharge test.
Each cycle of a cell yields one impedance spectrum (Re(Z), -Im(Z) over ~30-40
frequencies) and one discharge capacity; a model learns spectrum -> SOH and is
evaluated by **leave-one-cell-out (LOSO)** cross-validation.

Methodological anchors: Zhang et al. 2020 (GPR + ARD on EIS) and Jones et al.
2022 (XGBoost ensembles). See `docs/CONTEXT.md` for the full background.

## 1. Setup

```bash
conda create -n eis-ml python=3.12
conda activate eis-ml
pip install -r requirements.txt      # installs the eis_ml package in editable mode + dev tools
pytest                                # 15 unit tests, ~3 s, no data needed
```

## 2. Data

Download the dataset and unpack it into `data/` (not tracked in git):
<https://drive.google.com/file/d/1oxKcQ_CtknQq9hlbonAoCJDUV-l3i-30/view?usp=share_link>

```
data/
  PEIS-HC-RT/            A1..A8, B1..B6   PEIS every cycle, EIS at Ns=1 and Ns=6, capacity at Ns=8
  PEIS-HC-RT-sparseEIS/  A1..A8           PEIS every ~10 cycles, EIS at Ns=5, capacity at Ns=3
  GEIS-HC-RT/            B7, B8           GEIS every cycle, EIS at Ns=1 and Ns=6, capacity at Ns=8
  Na_NoEIS/              A1, A2           no impedance columns; not usable by this pipeline
```

Each CSV is one cell exported from Bio-Logic EC-Lab (`.mpt` -> `.csv`, long
form, one row per time step). The loader only reads the columns it needs:
`Ns`, `cycle number`, `freq/Hz`, `Re(Z)/Ohm`, `-Im(Z)/Ohm`, `Capacity/mA.h`, `I/mA`.

Set `EIS_ML_DATA=/path/to/data` to read from somewhere else (for example a
Dropbox-synced folder or a download cache). That is the only hook needed for
cloud-hosted data: fetch the CSVs into a local directory and point the
variable at it.

## 3. Run an experiment

```bash
# GPR (Zhang-style isotropic kernel) on the default Ns=1+6 state vector, with ARD frequency weights
python experiments/run_loso.py --dataset PEIS-HC-RT --model gpr --ard

# XGBoost ensemble, charged-state (Ns=6) spectrum only
python experiments/run_loso.py --dataset PEIS-HC-RT --model xgb --ns 6

# Other protocols use the same command; the preset knows their Ns steps
python experiments/run_loso.py --dataset GEIS-HC-RT --model xgb
python experiments/run_loso.py --dataset PEIS-HC-RT-sparseEIS --model gpr

# Model parameters are passed through
python experiments/run_loso.py --dataset PEIS-HC-RT --model gpr --param subset_size=None
python experiments/run_loso.py --dataset PEIS-HC-RT --model xgb --param n_models=5 --param max_depth=6

# Multi-step forecasting (SOH h cycles ahead from today's spectrum)
python experiments/xgb_multistep.py --dataset PEIS-HC-RT
```

Every run writes to `results/<model>/<dataset>_ns<steps>/`:

| file | content |
|---|---|
| `predictions.csv` | one row per held-out (channel, cycle): `y_true`, `y_pred`, `y_std` |
| `per_cell.csv` | RMSE / MAE / R2 per held-out cell |
| `summary.json` | config, pooled scores, RMSE by SOH band, sigma coverage |
| `parity.png`, `calibration.png` | pooled parity plot; reliability curve and residual-vs-sigma |
| `ard_weights.csv/.png` | GPR with `--ard`: relevance of every (Ns, Re/Im, frequency) feature |
| `feature_importance.csv` | XGB: gain importance per feature |

## 4. Using the package directly

```python
from eis_ml import get_dataset, load_dataset, build_dataset, loso_folds
from eis_ml.models import MODELS

spec = get_dataset("PEIS-HC-RT", eis_ns=[6])   # inject the EIS step(s) to use
df = load_dataset(spec)                        # long-form rows, all cells, 'channel' column added
X, y = build_dataset(df, spec)                 # X: (channel, cycle) x (ns, part, freq); y: SOH

for cell, train, test in loso_folds(X):
    model = MODELS["xgb"].fit(X[train], y[train])
    mean, std = MODELS["xgb"].predict(model, X[test])
```

`X.columns` is a three-level index `(ns, part, freq)`, so any per-feature
result (ARD weights, tree importances) maps back to a frequency without
hardcoded offsets: `eis_ml.features.feature_table(X)` flattens it.

### Package layout

```
eis_ml/
  datasets.py   DatasetSpec (eis_ns, capacity_ns, freq_range, cells) and presets; get_dataset()
  data.py       load_cell / load_dataset / list_cells; DATA_DIR from EIS_ML_DATA
  features.py   eis_features (vectorised pivot), capacity_labels (SOH), build_dataset
  splits.py     loso_folds (default), temporal_split (diagnostic)
  models/       gpr.py (isotropic predictor + ARD diagnostic), xgb.py (ensemble); MODELS registry
  metrics.py    pooled / per-cell scores, RMSE by SOH band, sigma coverage
  plots.py      parity, per-cell trajectories, calibration, ARD weights, Nyquist, degradation
experiments/    run_loso.py, xgb_multistep.py
tests/          unit tests on synthetic data (no raw data required)
notebooks/      untracked scratch space; import eis_ml like any package
docs/           CONTEXT.md (orientation), plans/, summaries/
results/        tracked outputs of experiments; results/legacy/ holds pre-refactor numbers
```

## 5. Protocols and labels

All cells were cycled at ~25 C with high C-rates. Within a cycle the tester
runs numbered steps (`Ns`); each protocol places the EIS sweep and the
capacity-bearing discharge in different steps, which is exactly what a
`DatasetSpec` records.

| dataset | EIS mode | EIS steps | frequencies (after 0.2 Hz < f <= 20 kHz filter) | capacity step |
|---|---|---|---|---|
| PEIS-HC-RT | potentiostatic (~10 mV) | Ns=1 before charge (discharged), Ns=6 after charge (charged) | 37 + 33 -> 140 features | Ns=8 (3.75C CC discharge) |
| GEIS-HC-RT | galvanostatic (~100 mA) | Ns=1, Ns=6 | 33 + 33 -> 132 features | Ns=8 |
| PEIS-HC-RT-sparseEIS | potentiostatic | Ns=5, only every ~10th cycle | 29 -> 58 features | Ns=3 (CC discharge) |

Rules baked into `eis_ml.features`:

- **Label:** the maximum `Capacity/mA.h` inside the capacity step of that cycle,
  divided by the same cell's first available cycle (SOH as a fraction). A warning
  is raised if the step's median current is not negative (not a discharge).
- **One row per (channel, cycle).** Channels are never mixed; a cycle is kept only
  if every requested Ns step was measured in it. Duplicate sweeps collapse to their median.
- **Feature order:** for each requested Ns in turn, all Re(Z) then all -Im(Z),
  frequency ascending. Missing frequencies are NaN (GPR imputes to the column
  mean after standardisation; XGBoost handles NaN natively).
- **No target-derived features.** Capacity, energy or SOC never enter `X`.

Cycle 0 (formation) and the last cycle of every file are dropped by the loader.

## 6. Evaluation

- **LOSO is the only default split.** Random row-level k-fold leaks a cell into
  both sides and is not provided. `temporal_split` exists for within-cell
  forecasting diagnostics.
- **RMSE (in SOH units) is the headline.** Per-cell R2 is meaningless for cells
  that barely degraded (variance ~ 0), so the pooled R2 over all held-out points
  is reported next to the per-cell table.
- `summary.json` also gives RMSE by SOH band and the fraction of residuals inside
  1 and 2 predicted sigma, which says whether the uncertainty is honest.

### Current results (PEIS-HC-RT, 14 cells, Ns=1+6, 140 features)

| model | pooled RMSE | pooled MAE | pooled R2 | mean per-cell RMSE | RMSE (bias) for SOH < 0.4 | within 2σ |
|---|---|---|---|---|---|---|
| GPR (isotropic, Zhang) | 0.132 | 0.062 | 0.831 | 0.102 | 0.273 (+0.17) | 54% |
| XGBoost ensemble (Jones) | 0.042 | 0.028 | 0.983 | 0.037 | 0.049 (+0.01) | 25% |

SOH is a fraction of first-cycle capacity, so RMSE 0.04 means 4% of initial capacity. Both models are weakest on deeply degraded cycles; GPR reverts toward the training mean there.

XGB multi-step forecasting (SOH h cycles ahead from one spectrum, pooled R2): h=0: 0.983, h=1: 0.983, h=2: 0.983, h=5: 0.983, h=10: 0.982, h=20: 0.977, h=40: 0.977.

Full tables: `results/gpr/PEIS-HC-RT_ns1-6/` and `results/xgb/PEIS-HC-RT_ns1-6/`.

## 7. Workflow and git

- Prototype in `notebooks/` (untracked). When a result is worth keeping, turn it
  into a script under `experiments/` that imports `eis_ml`, or add a flag to
  `run_loso.py` if it is a variant of LOSO.
- After changing `eis_ml`, run `pytest`, then `graphify update .` (see `CLAUDE.md`).
- Tracked: `eis_ml/`, `experiments/`, `tests/`, `docs/`, `results/` (small CSV/JSON/PNG).
  Ignored: `data/`, `notebooks/`, `models/`, `plots/`, `mlruns/`.
