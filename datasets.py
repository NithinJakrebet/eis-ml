"""Which data to load and where each protocol keeps its EIS sweep and capacity.

Every Bio-Logic protocol in ``data/`` is a sequence of numbered steps (``Ns``).
Which step carries the impedance sweep and which one carries the discharge
capacity differs per protocol, so each dataset is described by a small
:class:`DatasetSpec`. The EIS step(s) are *injectable*: pass ``eis_ns`` to
:func:`get_dataset` (or ``--ns`` on the experiment CLI) to run the same model
on a different state of charge.

Layout on disk: ``DATA_DIR/<dataset>/<cell>.csv``, one long-form CSV per cell
(one row per time step; EIS rows carry ``freq/Hz``). ``DATA_DIR`` defaults to
``<repo>/data`` and can be pointed elsewhere with the ``EIS_ML_DATA``
environment variable. Cloud storage only needs a fetch step that mirrors
``<dataset>/<cell>.csv`` into a local folder; nothing downstream changes.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent
DATA_DIR = Path(os.environ.get("EIS_ML_DATA", REPO_ROOT / "data"))
FREQ_RANGE_DEFAULT = (0.2, 20_000.0)

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


@dataclass(frozen=True)
class DatasetSpec:
    """Where to find EIS spectra and capacity labels inside one protocol.

    name:        folder under ``DATA_DIR`` (one CSV per cell inside it).
    eis_ns:      Ns step(s) whose impedance sweep forms the feature vector.
                 Several steps are concatenated in the order given.
    capacity_ns: discharge step whose ``Capacity/mA.h`` is the SOH label.
    freq_range:  keep frequencies ``lo < f <= hi`` (Hz); trims glitchy edges.
    cells:       restrict to these cells; ``None`` means every CSV in the folder.
    """

    name: str
    eis_ns: tuple[int, ...]
    capacity_ns: int
    freq_range: tuple[float, float] = FREQ_RANGE_DEFAULT
    cells: tuple[str, ...] | None = None

    def __post_init__(self) -> None:
        if not self.eis_ns:
            raise ValueError("eis_ns must name at least one Ns step")
        object.__setattr__(self, "eis_ns", tuple(int(n) for n in self.eis_ns))

    @property
    def tag(self) -> str:
        """Short id used for result folders, e.g. ``PEIS-HC-RT_ns1-6``."""
        return f"{self.name}_ns{'-'.join(map(str, self.eis_ns))}"

    def with_ns(self, *eis_ns: int) -> DatasetSpec:
        """Same protocol, different EIS step(s)."""
        return replace(self, eis_ns=tuple(eis_ns))


# Ns semantics per protocol (see docs/CONTEXT.md):
#   PEIS-HC-RT / GEIS-HC-RT : Ns=1 EIS before charge (discharged rest),
#                             Ns=6 EIS after charge (charged rest), Ns=8 CC discharge.
#   PEIS-HC-RT-sparseEIS    : Ns=5 EIS (every ~10 cycles), Ns=3 CC discharge.
#   Na_NoEIS                : no impedance columns at all; not usable here.
DATASETS: dict[str, DatasetSpec] = {
    "PEIS-HC-RT": DatasetSpec("PEIS-HC-RT", eis_ns=(1, 6), capacity_ns=8),
    "GEIS-HC-RT": DatasetSpec("GEIS-HC-RT", eis_ns=(1, 6), capacity_ns=8),
    "PEIS-HC-RT-sparseEIS": DatasetSpec("PEIS-HC-RT-sparseEIS", eis_ns=(5,), capacity_ns=3),
}


def get_dataset(
    name: str,
    eis_ns=None,
    capacity_ns: int | None = None,
    freq_range=None,
    cells=None,
) -> DatasetSpec:
    """Look up a preset by folder name and override any field.

    Unknown names are allowed as long as ``eis_ns`` and ``capacity_ns`` are
    given, so a new data folder needs no code change to be used.
    """
    base = DATASETS.get(name)
    if base is None:
        if eis_ns is None or capacity_ns is None:
            raise KeyError(
                f"Unknown dataset {name!r}; known: {sorted(DATASETS)}. "
                "Pass eis_ns and capacity_ns to describe a new one."
            )
        base = DatasetSpec(name, eis_ns=tuple(eis_ns), capacity_ns=capacity_ns)
    changes = {}
    if eis_ns is not None:
        changes["eis_ns"] = tuple(eis_ns)
    if capacity_ns is not None:
        changes["capacity_ns"] = int(capacity_ns)
    if freq_range is not None:
        changes["freq_range"] = (float(freq_range[0]), float(freq_range[1]))
    if cells is not None:
        changes["cells"] = tuple(cells)
    return replace(base, **changes) if changes else base


# --------------------------------------------------------------------------
# Loading
# --------------------------------------------------------------------------

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
