"""Train/test splits over a (channel, cycle)-indexed feature matrix.

Leave-one-cell-out (LOSO) is the default: it is the only split that tests
generalisation to an unseen cell. Random row-level k-fold leaks the same
cell into both sides and is deliberately not provided.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
import pandas as pd


def channels(X: pd.DataFrame) -> np.ndarray:
    return X.index.get_level_values("channel").to_numpy()


def loso_folds(X: pd.DataFrame, cells=None) -> Iterator[tuple[str, np.ndarray, np.ndarray]]:
    """Yield ``(held_out_cell, train_mask, test_mask)`` for every cell.

    ``cells`` limits which cells are held out (training still uses all others).
    """
    ch = channels(X)
    held_out = list(cells) if cells is not None else list(pd.unique(ch))
    for cell in held_out:
        test = ch == cell
        if not test.any():
            raise ValueError(f"Cell {cell!r} has no rows in X")
        yield cell, ~test, test


def temporal_split(X: pd.DataFrame, train_frac: float = 0.6) -> tuple[np.ndarray, np.ndarray]:
    """Within each cell, train on the earliest ``train_frac`` of cycles.

    Diagnostic split for "future cycles of a known cell" questions; it does
    not test cross-cell generalisation.
    """
    cycles = X.index.get_level_values("cycle").to_numpy()
    ch = channels(X)
    train = np.zeros(len(X), dtype=bool)
    for cell in pd.unique(ch):
        rows = np.flatnonzero(ch == cell)
        order = rows[np.argsort(cycles[rows], kind="stable")]
        train[order[: int(len(order) * train_frac)]] = True
    return train, ~train
