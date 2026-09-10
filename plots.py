"""Figures for LOSO results and quick EDA. Every function returns a Figure.

Result plots read the predictions table (``channel, cycle, y_true, y_pred,
y_std``) written by ``experiments/run_loso.py``.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import norm

# Categorical slots in fixed order (colorblind-checked); never cycled past 4 here.
SERIES = ("#2a78d6", "#eb6834", "#1baf7a", "#eda100")
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#d9d8d3"


def style_axes(ax):
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(GRID)
    ax.tick_params(colors=MUTED, labelsize=9)
    return ax


def parity(preds: pd.DataFrame, title: str = "LOSO parity") -> plt.Figure:
    """Predicted vs actual SOH for every held-out point, with pooled scores."""
    from metrics import pooled

    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    s = pooled(preds)
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    lo, hi = min(yt.min(), yp.min()), max(yt.max(), yp.max())
    ax.plot([lo, hi], [lo, hi], color=MUTED, lw=1, ls="--", zorder=1)
    ax.scatter(yt, yp, s=12, alpha=0.45, color=SERIES[0], edgecolors="none", zorder=2)
    ax.set_xlabel("Actual SOH")
    ax.set_ylabel("Predicted SOH")
    ax.set_title(f"{title}\npooled RMSE = {s['rmse']:.3f}   R² = {s['r2']:.3f}   n = {s['n']}", fontsize=10)
    ax.set_aspect("equal", adjustable="box")
    style_axes(ax)
    fig.tight_layout()
    return fig


def capacity_vs_cycle(preds: pd.DataFrame, ncols: int = 4) -> plt.Figure:
    """Small multiples: actual and predicted SOH along cycles, one panel per cell."""
    cells = list(pd.unique(preds["channel"]))
    nrows = int(np.ceil(len(cells) / ncols))
    fig, axes = plt.subplots(nrows, ncols, figsize=(3.6 * ncols, 2.8 * nrows), sharey=True, squeeze=False)
    for ax, cell in zip(axes.flat, cells):
        g = preds[preds["channel"] == cell].sort_values("cycle")
        c, yt, yp, sd = g["cycle"], g["y_true"], g["y_pred"], g["y_std"]
        ax.fill_between(c, yp - 2 * sd, yp + 2 * sd, color=SERIES[1], alpha=0.18, lw=0, label="±2σ")
        ax.plot(c, yt, color=SERIES[0], lw=2, label="actual")
        ax.plot(c, yp, color=SERIES[1], lw=2, ls="--", label="predicted")
        rmse = float(np.sqrt(np.mean((yt - yp) ** 2)))
        ax.set_title(f"{cell}   RMSE {rmse:.3f}", fontsize=10, loc="left")
        style_axes(ax)
    for ax in axes.flat[len(cells):]:
        ax.set_visible(False)
    for ax in axes[-1]:
        ax.set_xlabel("Cycle")
    for ax in axes[:, 0]:
        ax.set_ylabel("SOH")
    axes[0, 0].legend(frameon=False, fontsize=8, loc="lower left")
    fig.tight_layout()
    return fig


def calibration(preds: pd.DataFrame, title: str = "") -> plt.Figure:
    """Reliability curve and residual-vs-sigma scatter for the predicted uncertainty."""
    yt, yp = preds["y_true"].to_numpy(), preds["y_pred"].to_numpy()
    sd = preds["y_std"].clip(lower=1e-9).to_numpy()
    z = (yt - yp) / sd
    levels = np.linspace(0.05, 0.95, 19)
    empirical = [np.mean(np.abs(z) <= norm.ppf(0.5 + L / 2)) for L in levels]

    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 4.2))
    a.plot([0, 1], [0, 1], color=MUTED, lw=1, ls="--", label="perfect")
    a.plot(levels, empirical, "o-", color=SERIES[0], lw=2, ms=5, label="empirical")
    a.set_xlabel("Nominal coverage")
    a.set_ylabel("Empirical coverage")
    a.set_title("Reliability", fontsize=10, loc="left")
    a.legend(frameon=False, fontsize=8)
    style_axes(a)

    lim = float(max(sd.max(), np.abs(yt - yp).max()))
    b.plot([0, lim], [0, lim], color=MUTED, lw=1, ls="--", label="|resid| = σ")
    b.plot([0, lim / 2], [0, lim], color=MUTED, lw=0.8, ls=":", label="|resid| = 2σ")
    b.scatter(sd, np.abs(yt - yp), s=10, alpha=0.4, color=SERIES[0], edgecolors="none")
    b.set_xlabel("Predicted σ")
    b.set_ylabel("|residual|")
    b.set_title("Residual vs uncertainty", fontsize=10, loc="left")
    b.legend(frameon=False, fontsize=8)
    style_axes(b)
    if title:
        fig.suptitle(title, fontsize=10)
    fig.tight_layout()
    return fig


def ard_weights(weights: pd.DataFrame, title: str = "ARD relevance") -> plt.Figure:
    """Relevance vs frequency, one line per (Ns, Re/Im). ``weights`` has ns, part, freq, w_mean, w_std."""
    fig, ax = plt.subplots(figsize=(9, 4.2))
    ns_order = list(pd.unique(weights["ns"]))  # colour = Ns step, line style = Re / Im
    n_series = 0
    for ns in ns_order:
        color = SERIES[ns_order.index(ns) % len(SERIES)]
        for part, ls in (("re", "-"), ("im", "--")):
            g = weights[(weights["ns"] == ns) & (weights["part"] == part)].sort_values("freq")
            if g.empty:
                continue
            n_series += 1
            ax.plot(g["freq"], g["w_mean"], ls, color=color, lw=2, ms=4, marker="o",
                    label=f"Ns {ns} {'Re(Z)' if part == 're' else '-Im(Z)'}")
            ax.fill_between(g["freq"], g["w_mean"] - g["w_std"], g["w_mean"] + g["w_std"],
                            color=color, alpha=0.12, lw=0)
    ax.set_xscale("log")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Relevance  exp(−ℓ)")
    ax.set_title(title, fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8, ncol=min(n_series, 4))
    style_axes(ax)
    fig.tight_layout()
    return fig


def nyquist(df: pd.DataFrame, ns: int, cycles=None, title: str = "") -> plt.Figure:
    """EDA: Nyquist plot of one Ns step for a few cycles of one cell (``df`` = one cell)."""
    sweep = df[(df["Ns"] == ns) & (df["freq/Hz"] > 0)]
    cycles = list(cycles) if cycles is not None else sorted(sweep["cycle number"].unique())[::25]
    cmap = plt.get_cmap("Blues")
    fig, ax = plt.subplots(figsize=(5.5, 5))
    for k, cyc in enumerate(cycles):
        g = sweep[sweep["cycle number"] == cyc].sort_values("freq/Hz")
        ax.plot(g["Re(Z)/Ohm"], g["-Im(Z)/Ohm"], "o-", ms=3, lw=1,
                color=cmap(0.35 + 0.6 * k / max(len(cycles) - 1, 1)), label=f"cycle {int(cyc)}")
    ax.set_xlabel("Re(Z) / Ω")
    ax.set_ylabel("−Im(Z) / Ω")
    ax.set_aspect("equal", adjustable="datalim")
    ax.set_title(title or f"Nyquist, Ns {ns}", fontsize=10, loc="left")
    ax.legend(frameon=False, fontsize=8)
    style_axes(ax)
    fig.tight_layout()
    return fig


def degradation(labels: pd.DataFrame) -> plt.Figure:
    """EDA: SOH vs cycle for every cell (``labels`` from ``capacity_labels``)."""
    fig, ax = plt.subplots(figsize=(7, 4))
    cells = list(labels.index.get_level_values("channel").unique())
    cmap = plt.get_cmap("Blues")
    for k, cell in enumerate(cells):
        s = labels.xs(cell, level="channel")["soh"]
        ax.plot(s.index, s.values, lw=1.5, color=cmap(0.35 + 0.6 * k / max(len(cells) - 1, 1)))
        ax.annotate(cell, (s.index[-1], s.values[-1]), fontsize=7, color=MUTED, xytext=(3, 0),
                    textcoords="offset points", va="center")
    ax.set_xlabel("Cycle")
    ax.set_ylabel("SOH")
    ax.set_title("Capacity retention per cell", fontsize=10, loc="left")
    style_axes(ax)
    fig.tight_layout()
    return fig
