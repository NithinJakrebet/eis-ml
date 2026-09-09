"""Gaussian process regression for SOH, following Zhang et al. (2020).

Predictor (``fit``): isotropic squared-exponential kernel with a learned
Gaussian noise term, the same model as Zhang's ``Multi_T_EIS_Capacity_GPR.m``
(``covSEiso`` + ``likGauss``). Inputs are z-scored with training statistics.

Diagnostic (``fit_ard``): a separate ARD kernel with one length scale per
feature, used only to rank frequencies (Zhang's ``ARD_GPR.m``). It is not
the predictor.
"""

from __future__ import annotations

import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel


def _standardize_fit(X):
    X = np.asarray(X, dtype=float)
    mu = np.nanmean(X, axis=0)
    sig = np.nanstd(X, axis=0)
    mu = np.where(np.isnan(mu), 0.0, mu)
    sig = np.where(sig < 1e-12, 1.0, sig)
    return _standardize_apply(X, mu, sig), mu, sig


def _standardize_apply(X, mu, sig):
    # Missing cells become 0 (the column mean) so sparse EIS grids still work.
    return np.nan_to_num((np.asarray(X, dtype=float) - mu) / sig, nan=0.0)


def _stratified_subsample(X, y, n, seed):
    """Pick ~n rows spread evenly over SOH deciles."""
    rng = np.random.RandomState(seed)
    edges = np.quantile(y, np.linspace(0, 1, 11)[1:-1])
    bins = np.digitize(y, edges)
    chosen = []
    for b in np.unique(bins):
        rows = np.flatnonzero(bins == b)
        rng.shuffle(rows)
        chosen.extend(rows[: max(1, round(n * rows.size / len(y)))])
    chosen = np.asarray(chosen[:n])
    return X[chosen], y[chosen]


def _kernel(length_scale):
    return (
        ConstantKernel(1.0, (1e-3, 1e3))
        * RBF(length_scale=length_scale, length_scale_bounds=(1e-2, 1e2))
        + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5, 1e1))
    )


def _fit_gp(kernel, Xs, y, *, subset_size, n_restarts, random_state, alpha):
    """Optimise hyperparameters (optionally on a subset), then condition on all rows."""
    if subset_size and subset_size < len(y):
        Xsub, ysub = _stratified_subsample(Xs, y, subset_size, random_state)
        opt = GaussianProcessRegressor(
            kernel=kernel, alpha=alpha, normalize_y=True,
            n_restarts_optimizer=n_restarts, random_state=random_state,
        ).fit(Xsub, ysub)
        return GaussianProcessRegressor(
            kernel=opt.kernel_, alpha=alpha, normalize_y=True, optimizer=None,
        ).fit(Xs, y)
    return GaussianProcessRegressor(
        kernel=kernel, alpha=alpha, normalize_y=True,
        n_restarts_optimizer=n_restarts, random_state=random_state,
    ).fit(Xs, y)


def fit(X, y, subset_size=500, n_restarts=5, random_state=42, alpha=1e-10):
    """Zhang-style isotropic GPR predictor.

    subset_size: optimise the 3 kernel hyperparameters on a stratified subset
    of this many rows, then condition the posterior on the full training set.
    Exact-GP optimisation is O(n^3) per likelihood evaluation, so this is
    much faster than optimising on ~2500 rows and gives near-identical
    predictions. ``None`` optimises on everything.
    """
    y = np.asarray(y, dtype=float)
    Xs, mu, sig = _standardize_fit(X)
    model = _fit_gp(
        _kernel(1.0), Xs, y,
        subset_size=subset_size, n_restarts=n_restarts, random_state=random_state, alpha=alpha,
    )
    return {"model": model, "mu": mu, "sig": sig}


def predict(bundle, X):
    """Posterior mean and standard deviation of SOH."""
    Xs = _standardize_apply(X, bundle["mu"], bundle["sig"])
    return bundle["model"].predict(Xs, return_std=True)


def fit_ard(X, y, subset_size=300, n_restarts=1, random_state=42, alpha=1e-10):
    """ARD diagnostic: one length scale per feature, fit on a subset for speed."""
    y = np.asarray(y, dtype=float)
    Xs, mu, sig = _standardize_fit(X)
    model = _fit_gp(
        _kernel(np.ones(Xs.shape[1])), Xs, y,
        subset_size=subset_size, n_restarts=n_restarts, random_state=random_state, alpha=alpha,
    )
    return {"model": model, "mu": mu, "sig": sig}


def ard_weights(bundle) -> np.ndarray:
    """Per-feature relevance ``exp(-length_scale)``: larger means more relevant."""
    rbf = bundle["model"].kernel_.k1.k2  # (Constant * RBF) + White
    return np.exp(-np.atleast_1d(rbf.length_scale))
