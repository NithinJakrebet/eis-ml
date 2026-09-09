"""eis_ml: predict Li-ion battery state of health (SOH) from EIS spectra.

Typical use::

    from eis_ml import get_dataset, load_dataset, build_dataset, loso_folds
    from eis_ml.models import MODELS

    spec = get_dataset("PEIS-HC-RT", eis_ns=[6])     # inject which Ns step(s) to use
    df = load_dataset(spec)                          # long-form rows for every cell
    X, y = build_dataset(df, spec)                   # one row per (channel, cycle)
    for cell, train, test in loso_folds(X):
        model = MODELS["gpr"].fit(X[train], y[train])
        mean, std = MODELS["gpr"].predict(model, X[test])
"""

from .datasets import DATASETS, DatasetSpec, get_dataset
from .data import DATA_DIR, list_cells, load_cell, load_dataset
from .features import build_dataset, capacity_labels, eis_features
from .splits import loso_folds, temporal_split

__all__ = [
    "DATASETS",
    "DATA_DIR",
    "DatasetSpec",
    "build_dataset",
    "capacity_labels",
    "eis_features",
    "get_dataset",
    "list_cells",
    "load_cell",
    "load_dataset",
    "loso_folds",
    "temporal_split",
]
