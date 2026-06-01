
import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.append("scripts")

from data_pipeline import loso
from feature_engineering import build_model_input
from algorithms import gpr
import evaluate

# ---------------------------------------------------------------------------
# Dataset configuration (plain kwargs — edit these for other datasets).
#   PEIS-HC-RT   : ns_states=[1, 6],  capacity_ns=8   (37 + 33 EIS freqs)
#   GEIS-HC-RT   : ns_states=[1, 6],  capacity_ns=8   (33 + 33 EIS freqs)
#   PEIS-sparseEIS: ns_states=[5],    capacity_ns=8   (single EIS step)
# ---------------------------------------------------------------------------
data_folder = "PEIS-HC-RT"
cv_method = "LOSO"

ns_states = [1, 6]
freq_range = (0.2, 20000)
capacity_ns = 8

CHANNELS = [
    'A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8',
    'B1', 'B2', 'B3', 'B4', 'B5', 'B6'
]

# GPR predictor params (Zhang-faithful isotropic SE + learned noise)
gpr_params = {
    'alpha': 1e-10,
    'normalize_y': True,
    'n_restarts_optimizer': 5,
    'random_state': 42,
}
# ARD diagnostic is fit on a stratified subset for speed (weights only)
ard_subset_size = 300

print("Configuration loaded:")
print(f"  Data: {data_folder}")
print(f"  CV Method: {cv_method}")
print(f"  EIS Ns states: {ns_states}  |  capacity Ns: {capacity_ns}")

data_path = Path("data") / data_folder

results = []          # per-fold (cell, rmse, mae, r2)
models = {}
W = []                # per-fold ARD weight vectors (aligned to layout columns)
layout_ref = None
y_true_all, y_pred_all = [], []

print("Starting training loop")
for i, test_cell in enumerate(CHANNELS):
    train_cells = [c for c in CHANNELS if c != test_cell]
    df_train, df_test = loso(data_path, train_cells, [test_cell])

    X_train, y_train, layout = build_model_input(
        df_train, ns_states=ns_states, freq_range=freq_range,
        capacity_ns=capacity_ns, return_layout=True
    )
    X_test, y_test = build_model_input(
        df_test, ns_states=ns_states, freq_range=freq_range,
        capacity_ns=capacity_ns
    )
    if layout_ref is None:
        layout_ref = layout

    # Prediction model: isotropic SE + learned noise, fit on full training data
    model = gpr.train_capacity_gpr(X_train, y_train, gpr_params=gpr_params)
    models[test_cell] = model

    y_pred, y_std = gpr.predict(model, X_test)
    rmse, r2, mse, mae = evaluate.evaluate_model(y_test, y_pred)
    results.append((test_cell, rmse, mae, r2))

    y_true_all.append(np.asarray(y_test))
    y_pred_all.append(np.asarray(y_pred))

    # Separate ARD diagnostic purely for per-frequency relevance weights
    ard = gpr.train_ard_diagnostic(X_train, y_train, gpr_params=gpr_params,
                                   subset_size=ard_subset_size)
    W.append(gpr.ard_frequency_weights(ard))

print(f"Completed {len(results)} folds")

# ---------------------------------------------------------------------------
# Metrics: per-cell RMSE/MAE/R2 plus a pooled R2 (honest denominator).
# Per-cell R2 is unreliable for low-degradation cells (tiny Var(y_test)),
# so we also report R2 pooled across every test prediction.
# ---------------------------------------------------------------------------
df = pd.DataFrame(results, columns=["cell", "rmse", "mae", "r2"]).sort_values("cell")

y_true_all = np.concatenate(y_true_all)
y_pred_all = np.concatenate(y_pred_all)
from sklearn.metrics import r2_score, mean_squared_error
pooled_r2 = r2_score(y_true_all, y_pred_all)
pooled_rmse = np.sqrt(mean_squared_error(y_true_all, y_pred_all))

print("\nPer-cell performance:")
print(df.to_string(index=False))
print(f"\nMean per-cell RMSE: {df['rmse'].mean():.4f}  MAE: {df['mae'].mean():.4f}")
print(f"Pooled (all test points) R2: {pooled_r2:.4f}  |  RMSE: {pooled_rmse:.4f}")

# Pooled parity plot
out_dir = Path("results") / "gpr" / "outputs"
out_dir.mkdir(parents=True, exist_ok=True)
fig, ax = plt.subplots(figsize=(6, 6))
ax.scatter(y_true_all, y_pred_all, s=10, alpha=0.4, color="#2d6cdf")
lims = [min(y_true_all.min(), y_pred_all.min()), max(y_true_all.max(), y_pred_all.max())]
ax.plot(lims, lims, "k--", lw=1)
ax.set_xlabel("Actual SOH")
ax.set_ylabel("Predicted SOH")
ax.set_title(f"LOSO parity ({data_folder})\npooled R2={pooled_r2:.3f}  RMSE={pooled_rmse:.3f}")
fig.tight_layout()
parity_path = out_dir / f"gpr_loso_parity_{data_folder}.png"
fig.savefig(parity_path, dpi=200)
print(f"Saved parity plot -> {parity_path}")

# ---------------------------------------------------------------------------
# ARD frequency weights, grouped generically from the feature layout
# (no hardcoded 37/33 split — driven by layout['columns']).
# ---------------------------------------------------------------------------
W = np.vstack(W)                      # (n_folds, n_features)
w_mean = W.mean(axis=0)
w_std = W.std(axis=0)

cols = layout_ref["columns"]          # list of (ns, freq, 'Re'|'Im')
wdf = pd.DataFrame(cols, columns=["ns_state", "freq_hz", "comp"])
wdf["w_mean"] = w_mean
wdf["w_std"] = w_std

# Pivot Re/Im onto one row per (ns, freq)
re = wdf[wdf["comp"] == "Re"].rename(columns={"w_mean": "w_re_mean", "w_std": "w_re_std"})
im = wdf[wdf["comp"] == "Im"].rename(columns={"w_mean": "w_im_mean", "w_std": "w_im_std"})
ard_df = re[["ns_state", "freq_hz", "w_re_mean", "w_re_std"]].merge(
    im[["ns_state", "freq_hz", "w_im_mean", "w_im_std"]],
    on=["ns_state", "freq_hz"], how="outer"
)
ard_df["w_avg_mean"] = (ard_df["w_re_mean"] + ard_df["w_im_mean"]) / 2.0
ard_df["w_avg_std"] = np.sqrt((ard_df["w_re_std"] ** 2 + ard_df["w_im_std"] ** 2) / 2.0)
ard_df = ard_df.sort_values("w_avg_mean", ascending=False).reset_index(drop=True)

print("\nARD Frequency Weights (sorted by importance)")
print("=" * 72)
print(f"{'Rank':<6}{'Ns':<6}{'Freq(Hz)':<12}{'Re wt':<12}{'Im wt':<12}{'Avg wt':<10}")
print("-" * 72)
for idx, row in ard_df.iterrows():
    print(f"{idx+1:<6}{int(row['ns_state']):<6}{row['freq_hz']:<12.2f}"
          f"{row['w_re_mean']:<12.4f}{row['w_im_mean']:<12.4f}{row['w_avg_mean']:<10.4f}")

ard_csv = out_dir / f"gpr_loso_ard_weights_{data_folder}.csv"
ard_df.to_csv(ard_csv, index=False)
print(f"\nSaved ARD weights -> {ard_csv}")
