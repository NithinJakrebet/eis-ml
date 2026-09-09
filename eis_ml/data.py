"""Raw cell data access.

Layout on disk: ``DATA_DIR/<dataset>/<cell>.csv`` where each CSV is one
Bio-Logic ``.mpt`` export (long form: one row per time step, EIS rows carry
``freq/Hz``). ``DATA_DIR`` defaults to ``<repo>/data`` and can be pointed
elsewhere with the ``EIS_ML_DATA`` environment variable.

Cloud storage: everything downstream only needs :func:`load_cell` to return
a DataFrame. To pull from Dropbox (or any remote), add a fetch step that
downloads ``<dataset>/<cell>.csv`` into a local cache directory and set
``EIS_ML_DATA`` to that cache. Nothing else in the package has to change.
"""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd

from .datasets import DatasetSpec

REPO_ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = Path(os.environ.get("EIS_ML_DATA", REPO_ROOT / "data"))

# Columns the pipeline needs. Bio-Logic exports "-Im(Z)/Ohm" as "#NAME?"
# because spreadsheet software mangles the leading minus sign.
_RENAME = {"#NAME?": "-Im(Z)/Ohm"}
BASE_COLUMNS = (
    "Ns",
    "cycle number",
    "freq/Hz",
    "Re(Z)/Ohm",
    "-Im(Z)/Ohm",
    "Capacity/mA.h",
    "I/mA",
)


def cell_path(dataset: str, cell: str) -> Path:
    return DATA_DIR / dataset / f"{cell}.csv"


def list_cells(dataset: str) -> list[str]:
    """Cell ids (CSV stems) available for a dataset folder."""
    folder = DATA_DIR / dataset
    if not folder.is_dir():
        raise FileNotFoundError(f"No data folder {folder}. Set EIS_ML_DATA or download the data.")
    return sorted(p.stem for p in folder.glob("*.csv"))


def load_cell(
    dataset: str,
    cell: str,
    columns=BASE_COLUMNS,
    trim_cycles: bool = True,
) -> pd.DataFrame:
    """Load one cell's long-form CSV and tag every row with its ``channel``.

    columns:     which CSV columns to keep (extra per-cycle variables can be
                 requested here for future feature sets). Missing columns
                 are ignored so protocols without EIS still load.
    trim_cycles: drop cycle 0 (formation) and the last cycle, which is
                 usually cut short when the run stops.
    """
    path = cell_path(dataset, cell)
    wanted = set(columns) | {k for k, v in _RENAME.items() if v in columns}
    df = pd.read_csv(path, usecols=lambda c: c in wanted, low_memory=False)
    df = df.rename(columns=_RENAME)
    for col in ("Re(Z)/Ohm", "-Im(Z)/Ohm", "freq/Hz"):
        if col in df:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    df.insert(0, "channel", cell)

    if trim_cycles:
        cyc = df["cycle number"]
        df = df[(cyc > 0) & (cyc < cyc.max())]
    return df.reset_index(drop=True)


def load_dataset(spec: DatasetSpec, columns=BASE_COLUMNS) -> pd.DataFrame:
    """Concatenate every cell of a dataset (all CSVs, or ``spec.cells``)."""
    cells = list(spec.cells) if spec.cells else list_cells(spec.name)
    if not cells:
        raise FileNotFoundError(f"No CSV files under {DATA_DIR / spec.name}")
    frames = [load_cell(spec.name, c, columns=columns) for c in cells]
    return pd.concat(frames, ignore_index=True)
