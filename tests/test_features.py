import numpy as np
import pandas as pd
import pytest

from datasets import DatasetSpec
from features import build_dataset, capacity_labels, eis_features, feature_table

NS1_FREQS = [0.5, 5.0, 50.0]
NS6_FREQS = [5.0, 50.0]


def _eis_rows(channel, cycle, ns, freqs, re_offset=0.0):
    return [
        {
            "channel": channel, "cycle number": cycle, "Ns": ns, "freq/Hz": f,
            "Re(Z)/Ohm": ns + f + re_offset, "-Im(Z)/Ohm": -f, "Capacity/mA.h": 0.0, "I/mA": 0.0,
        }
        for f in freqs
    ]


def _capacity_rows(channel, cycle, cap):
    # accumulating discharge capacity: the cycle's value is the maximum
    return [
        {"channel": channel, "cycle number": cycle, "Ns": 8, "freq/Hz": 0.0,
         "Re(Z)/Ohm": 0.0, "-Im(Z)/Ohm": 0.0, "Capacity/mA.h": c, "I/mA": -1500.0}
        for c in (cap * 0.5, cap)
    ]


@pytest.fixture
def df():
    rows = []
    for ch in ("A1", "A2"):
        for cyc in (1, 2, 3):
            rows += _eis_rows(ch, cyc, 1, NS1_FREQS)
            if not (ch == "A2" and cyc == 3):  # A2 cycle 3 has no charged-state sweep
                rows += _eis_rows(ch, cyc, 6, NS6_FREQS)
            rows += _capacity_rows(ch, cyc, 1000.0 * (1 - 0.1 * (cyc - 1)))
    # duplicate sweep for A1 cycle 1 at Ns=1 with shifted Re -> median must be used
    rows += _eis_rows("A1", 1, 1, NS1_FREQS, re_offset=2.0)
    rows += _eis_rows("A1", 1, 1, NS1_FREQS, re_offset=4.0)
    # out-of-range frequencies that must be dropped, and one exactly at the upper edge (kept)
    rows += _eis_rows("A1", 1, 1, [0.1, 30000.0])
    rows += _eis_rows("A1", 1, 6, [20000.0])
    # a cycle lacking a capacity step entirely
    rows += _eis_rows("A1", 4, 1, NS1_FREQS) + _eis_rows("A1", 4, 6, NS6_FREQS)
    return pd.DataFrame(rows)


def test_layout_and_rows(df):
    X = eis_features(df, eis_ns=[1, 6])
    grids = {1: NS1_FREQS, 6: NS6_FREQS + [20000.0]}
    expected = [(ns, part, f) for ns in (1, 6) for part in ("re", "im") for f in grids[ns]]
    assert list(X.columns) == expected
    assert X.index.names == ["channel", "cycle"]
    # A2 cycle 3 lacks Ns=6 -> dropped; A1 cycle 4 has both -> kept here
    assert set(X.index) == {("A1", 1), ("A1", 2), ("A1", 3), ("A1", 4), ("A2", 1), ("A2", 2)}


def test_median_over_duplicate_sweeps_and_nan_for_missing_freq(df):
    X = eis_features(df, eis_ns=[1, 6])
    # three sweeps with Re offsets 0, 2, 4 -> median offset 2
    assert X.loc[("A1", 1), (1, "re", 5.0)] == pytest.approx(1 + 5.0 + 2.0)
    assert X.loc[("A1", 2), (1, "re", 5.0)] == pytest.approx(1 + 5.0)
    # 20 kHz only measured on A1 cycle 1 -> NaN elsewhere
    assert np.isnan(X.loc[("A2", 1), (6, "im", 20000.0)])
    assert X.loc[("A1", 1), (6, "im", 20000.0)] == pytest.approx(-20000.0)


def test_single_injected_ns(df):
    X = eis_features(df, eis_ns=[6])
    assert sorted(set(X.columns.get_level_values("ns"))) == [6]
    assert ("A2", 3) not in X.index
    X1 = eis_features(df, eis_ns=[1])
    assert ("A2", 3) in X1.index  # Ns=1 alone does not need the charged sweep


def test_capacity_labels(df):
    lab = capacity_labels(df, capacity_ns=8)
    assert lab.loc[("A1", 1), "capacity"] == 1000.0
    assert lab.loc[("A1", 3), "soh"] == pytest.approx(0.8)
    assert lab.loc[("A2", 2), "soh"] == pytest.approx(0.9)


def test_capacity_warns_when_not_discharge(df):
    charge = df.copy()
    charge.loc[charge["Ns"] == 8, "I/mA"] = 1500.0
    with pytest.warns(UserWarning, match="discharge"):
        capacity_labels(charge, capacity_ns=8)


def test_build_dataset_aligns_and_drops(df):
    spec = DatasetSpec("synthetic", eis_ns=(1, 6), capacity_ns=8)
    X, y = build_dataset(df, spec)
    assert list(X.index) == list(y.index)
    assert ("A1", 4) not in X.index  # no capacity row
    assert ("A2", 3) not in X.index  # no Ns=6 sweep
    assert y.loc[("A1", 2)] == pytest.approx(0.9)
    assert X.shape == (5, 2 * len(NS1_FREQS) + 2 * (len(NS6_FREQS) + 1))


def test_errors_on_missing_steps(df):
    with pytest.raises(ValueError, match="No EIS rows"):
        eis_features(df, eis_ns=[3])
    with pytest.raises(ValueError, match="No rows at capacity_ns"):
        capacity_labels(df, capacity_ns=5)


def test_feature_table(df):
    X = eis_features(df, eis_ns=[1])
    tab = feature_table(X)
    assert list(tab.columns) == ["ns", "part", "freq"]
    assert len(tab) == X.shape[1]
