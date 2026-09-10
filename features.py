"""Turn long-form cell data into one feature row per (channel, cycle).

Feature matrix ``X``: a DataFrame indexed by ``(channel, cycle)`` with a
three-level column index ``(ns, part, freq)``. Column order is, for each
requested Ns step in turn, all ``re`` values then all ``im`` values with
frequency ascending, so ``X.columns`` *is* the feature layout and anything
that maps weights back to frequencies just reads it.

Label ``y``: SOH = discharge capacity at ``capacity_ns`` divided by the same
cell's capacity on its first available cycle.

Invariants kept from the original pipeline:
- never mix channels: every row is one (channel, cycle);
- keep a cycle only if every requested Ns step was measured in it;
- duplicate (cycle, freq) rows collapse to their median;
- capacity never enters the features.
"""

from __future__ import annotations

import warnings

import numpy as np
import pandas as pd

from datasets import DatasetSpec

PARTS = ("re", "im")
_EIS_COLS = {
    "cycle number": "cycle",
    "Ns": "ns",
    "freq/Hz": "freq",
    "Re(Z)/Ohm": "re",
    "-Im(Z)/Ohm": "im",
}


def eis_features(df: pd.DataFrame, eis_ns, freq_range=(0.2, 20_000.0)) -> pd.DataFrame:
    """Wide impedance matrix for the requested Ns steps.

    Frequencies are the union over all cells, so every cell shares one
    column grid; a cell missing a frequency gets NaN there (models impute
    or handle NaN natively).
    """
    if "channel" not in df:
        raise ValueError("df needs a 'channel' column; load it with datasets.load_cell")
    eis_ns = [int(n) for n in eis_ns]
    lo, hi = freq_range
    mask = df["Ns"].isin(eis_ns) & (df["freq/Hz"] > lo) & (df["freq/Hz"] <= hi)
    eis = df.loc[mask, ["channel", *_EIS_COLS]].rename(columns=_EIS_COLS)
    if eis.empty:
        raise ValueError(
            f"No EIS rows at Ns={eis_ns} in {freq_range} Hz. "
            f"Ns values present: {sorted(df['Ns'].dropna().unique().tolist())}"
        )
    eis["ns"] = eis["ns"].astype(int)

    long = eis.melt(
        id_vars=["channel", "cycle", "ns", "freq"], value_vars=list(PARTS), var_name="part"
    ).dropna(subset=["value"])
    wide = long.pivot_table(
        index=["channel", "cycle"], columns=["ns", "part", "freq"], values="value", aggfunc="median"
    )

    grids = {ns: sorted(long.loc[long["ns"] == ns, "freq"].unique()) for ns in eis_ns}
    missing = [ns for ns, g in grids.items() if not g]
    if missing:
        raise ValueError(f"No EIS rows for Ns={missing} after frequency filtering")
    columns = pd.MultiIndex.from_tuples(
        [(ns, part, f) for ns in eis_ns for part in PARTS for f in grids[ns]],
        names=["ns", "part", "freq"],
    )
    wide = wide.reindex(columns=columns)

    # A cycle counts only if every requested step has at least one point.
    complete = np.logical_and.reduce([wide[ns].notna().any(axis=1).values for ns in eis_ns])
    return wide[complete]


def capacity_labels(df: pd.DataFrame, capacity_ns: int) -> pd.DataFrame:
    """Per (channel, cycle) discharge capacity and SOH from one Ns step.

    Capacity accumulates within the step, so the cycle's capacity is the
    step's maximum. SOH normalises by the cell's first available cycle.
    """
    step = df[df["Ns"] == capacity_ns]
    if step.empty:
        raise ValueError(
            f"No rows at capacity_ns={capacity_ns}. "
            f"Ns values present: {sorted(df['Ns'].dropna().unique().tolist())}"
        )
    if "I/mA" in step and step["I/mA"].median() >= 0:
        warnings.warn(
            f"Ns={capacity_ns} does not look like a discharge step (median I/mA >= 0); "
            "check capacity_ns for this dataset.",
            stacklevel=2,
        )
    cap = step.groupby(["channel", "cycle number"])["Capacity/mA.h"].max()
    cap.index = cap.index.set_names(["channel", "cycle"])
    first = cap.groupby(level="channel").transform("first")
    return pd.DataFrame({"capacity": cap, "soh": cap / first})


def build_dataset(df: pd.DataFrame, spec: DatasetSpec) -> tuple[pd.DataFrame, pd.Series]:
    """``(X, y)`` aligned on ``(channel, cycle)`` for a dataset spec.

    Cycles lacking either the EIS sweep or the capacity step are dropped.
    """
    X = eis_features(df, spec.eis_ns, spec.freq_range)
    labels = capacity_labels(df, spec.capacity_ns)
    idx = X.index.intersection(labels.index)
    if len(idx) == 0:
        raise ValueError(
            f"No cycle has both EIS at Ns={list(spec.eis_ns)} and capacity at Ns={spec.capacity_ns}"
        )
    return X.loc[idx], labels.loc[idx, "soh"].rename("soh")


def feature_table(X: pd.DataFrame) -> pd.DataFrame:
    """Columns of ``X`` as a flat table (ns, part, freq), one row per feature."""
    return X.columns.to_frame(index=False)
