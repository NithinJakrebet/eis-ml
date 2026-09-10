# EIS-ML: Battery State of Health from Impedance Spectroscopy

Predict Li-ion battery **state of health (SOH)** from one electrochemical impedance
spectroscopy (EIS) sweep, evaluated by **leave-one-cell-out (LOSO)** cross-validation.
Anchored on Zhang et al. 2020 (GPR + ARD) and Jones et al. 2022 (XGBoost ensembles).
Background, data quirks and findings: [`docs/CONTEXT.md`](docs/CONTEXT.md).

## Setup

```bash
conda create -n eis-ml python=3.12 && conda activate eis-ml
pip install -r requirements.txt
pytest                      # unit tests on synthetic data, no raw data needed
```

Download the data (<https://drive.google.com/file/d/1oxKcQ_CtknQq9hlbonAoCJDUV-l3i-30/view?usp=share_link>)
into `data/<dataset>/<cell>.csv`, or set `EIS_ML_DATA=/path/to/data` to read from
elsewhere (a Dropbox-synced folder works as-is).

## Run

```bash
python experiments/run_loso.py --dataset PEIS-HC-RT --model gpr --ard   # Zhang-style GPR + ARD weights
python experiments/run_loso.py --dataset PEIS-HC-RT --model xgb --ns 6  # XGB on the charged-state sweep only
python experiments/run_loso.py --dataset GEIS-HC-RT --model xgb --param n_models=5
python experiments/xgb_multistep.py --dataset PEIS-HC-RT                # SOH h cycles ahead
```

Outputs land in `results/<model>/<dataset>_ns<steps>/`: `predictions.csv`,
`per_cell.csv`, `summary.json` (pooled scores, RMSE by SOH band, sigma coverage),
`parity.png`, `trajectories.png`, `calibration.png`, and `ard_weights.csv/.png`
(GPR with `--ard`) or `feature_importance.csv` (XGB).

## Layout

```
datasets.py     DatasetSpec presets (which Ns holds the EIS sweep / the capacity), CSV loading
features.py     one row per (channel, cycle); columns (ns, part, freq); SOH labels
splits.py       loso_folds (default), temporal_split (diagnostic)
algorithms/     gpr.py, xgb.py, both expose fit(X, y) -> bundle and predict(bundle, X) -> (mean, std)
metrics.py      pooled / per-cell scores, RMSE by SOH band, sigma coverage
plots.py        parity, trajectories, calibration, ARD weights, Nyquist, degradation
experiments/    run_loso.py, xgb_multistep.py
tests/          pytest
results/        tracked experiment outputs (results/legacy = pre-refactor)
notebooks/      untracked scratch; start with  import sys; sys.path.insert(0, "..")
```

Using the modules directly:

```python
from datasets import get_dataset, load_dataset
from features import build_dataset
from splits import loso_folds
from algorithms import MODELS

spec = get_dataset("PEIS-HC-RT", eis_ns=[6])     # the EIS step is injectable
X, y = build_dataset(load_dataset(spec), spec)   # X: (channel, cycle) x (ns, part, freq)
for cell, train, test in loso_folds(X):
    model = MODELS["xgb"].fit(X[train], y[train])
    mean, std = MODELS["xgb"].predict(model, X[test])
```

## Datasets

| dataset | cells | EIS steps | capacity step | features |
|---|---|---|---|---|
| PEIS-HC-RT | A1-A8, B1-B6 | Ns 1 (discharged), Ns 6 (charged) | Ns 8 | 37 + 33 freqs x Re/Im = 140 |
| GEIS-HC-RT | B7, B8 | Ns 1, Ns 6 | Ns 8 | 132 |
| PEIS-HC-RT-sparseEIS | A1-A8 | Ns 5, every ~10 cycles | Ns 3 | 58 |

SOH = max `Capacity/mA.h` in the capacity step, divided by the cell's first cycle.
Only frequencies in 0.2 Hz < f <= 20 kHz are kept; a cycle needs every requested Ns
step; duplicate sweeps collapse to the median; capacity never enters the features.

## Results (PEIS-HC-RT, 14-cell LOSO, Ns 1+6)

| model | pooled RMSE | pooled R2 | mean per-cell RMSE |
|---|---|---|---|
| GPR (isotropic, Zhang) | 0.132 | 0.83 | 0.10 |
| XGBoost ensemble (Jones) | 0.042 | 0.98 | 0.037 |

RMSE is in SOH units (0.04 = 4% of initial capacity) and is the headline metric;
per-cell R2 is meaningless for cells that barely degrade. Details and figures in
`results/gpr/PEIS-HC-RT_ns1-6/` and `results/xgb/PEIS-HC-RT_ns1-6/`.

## Conventions

- LOSO only; random row-level k-fold leaks cells and is not provided.
- Prototype in `notebooks/`, promote keepers to `experiments/*.py`.
- Run `pytest` and `graphify update .` after changing code (see `CLAUDE.md`).
- Not tracked: `data/`, `notebooks/`, `models/`, `plots/`, `mlruns/`.
