import numpy as np
import pandas as pd
import pytest

from algorithms import MODELS, gpr, xgb
from splits import loso_folds, temporal_split


@pytest.fixture
def toy():
    rng = np.random.RandomState(0)
    index = pd.MultiIndex.from_tuples(
        [(c, k) for c in ("A1", "A2", "B1") for k in range(1, 21)], names=["channel", "cycle"]
    )
    X = pd.DataFrame(rng.normal(size=(len(index), 6)), index=index)
    y = pd.Series(1 - 0.02 * index.get_level_values("cycle") + 0.05 * X[0], index=index)
    return X, y


def test_loso_folds_are_cell_disjoint(toy):
    X, _ = toy
    folds = list(loso_folds(X))
    assert [c for c, _, _ in folds] == ["A1", "A2", "B1"]
    for cell, train, test in folds:
        assert not (train & test).any()
        assert set(X.index[test].get_level_values("channel")) == {cell}
        assert cell not in set(X.index[train].get_level_values("channel"))
    assert [c for c, _, _ in loso_folds(X, cells=["B1"])] == ["B1"]


def test_temporal_split_uses_early_cycles(toy):
    X, _ = toy
    train, test = temporal_split(X, train_frac=0.6)
    cyc = X.index.get_level_values("cycle").to_numpy()
    assert train.sum() == 3 * 12
    assert cyc[train].max() == 12 and cyc[test].min() == 13


@pytest.mark.parametrize("name", ["gpr", "xgb"])
def test_model_roundtrip(toy, name):
    X, y = toy
    _, train, test = next(loso_folds(X))
    kw = {"n_restarts": 0} if name == "gpr" else {"n_models": 2, "n_estimators": 20}
    bundle = MODELS[name].fit(X[train], y[train], **kw)
    mean, std = MODELS[name].predict(bundle, X[test])
    assert mean.shape == std.shape == (test.sum(),)
    assert np.all(std >= 0)


def test_gpr_handles_nan_and_subset(toy):
    X, y = toy
    X = X.copy()
    X.iloc[::5, 2] = np.nan
    bundle = gpr.fit(X, y, subset_size=20, n_restarts=0)
    mean, _ = gpr.predict(bundle, X.iloc[:3])
    assert np.isfinite(mean).all()


def test_ard_weights_length(toy):
    X, y = toy
    bundle = gpr.fit_ard(X, y, subset_size=30, n_restarts=0)
    w = gpr.ard_weights(bundle)
    assert w.shape == (X.shape[1],)
    assert (w > 0).all()


def test_xgb_feature_importance(toy):
    X, y = toy
    bundle = xgb.fit(X, y, n_models=2, n_estimators=20)
    assert xgb.feature_importance(bundle).shape == (X.shape[1],)


def test_gpr_linear_mean_extrapolates():
    # SOH is linear in one feature; hold out the rows beyond the training range so the
    # zero-mean GP has to extrapolate (it reverts to the mean) while the linear mean follows the trend.
    rng = np.random.RandomState(1)
    index = pd.MultiIndex.from_tuples([("A1", k) for k in range(80)], names=["channel", "cycle"])
    X = pd.DataFrame(rng.normal(size=(80, 4)), index=index)
    X[0] = np.linspace(-2, 2, 80)
    y = pd.Series(1.0 - 0.15 * X[0] + 0.01 * X[1], index=index)
    train, test = (X[0] <= 0.8).to_numpy(), (X[0] > 0.8).to_numpy()
    zero = gpr.fit(X[train], y[train], n_restarts=0, mean="zero")
    linear = gpr.fit(X[train], y[train], n_restarts=0, mean="linear")
    rmse = lambda b: float(np.sqrt(np.mean((gpr.predict(b, X[test])[0] - y[test].to_numpy()) ** 2)))
    assert rmse(linear) < rmse(zero)
    assert linear["trend"] is not None and zero["trend"] is None
    with pytest.raises(ValueError):
        gpr.fit(X[train], y[train], mean="quadratic")
