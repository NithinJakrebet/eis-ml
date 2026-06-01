# Plan: Zhang-faithful GPR + injectable-Ns pipeline

## Context

Two motivations:

1. **GPR fidelity / negative R².** Our current GPR ([scripts/algorithms/gpr.py](../../Desktop/Rolston_Labs/eis-ml/scripts/algorithms/gpr.py)) diverges from Zhang's actual capacity model: we use a 140-dim **ARD** kernel with **fixed** near-zero noise (`WhiteKernel(1e-3,'fixed')`, `alpha=0`), hyperparameters fit on a **300-sample subset**. Zhang's capacity predictor is an **isotropic** SE (`covSEiso`, 2 hyperparams) with **learned** Gaussian noise (`sn≈0.1`) fit on all data; their ARD (`covSEard`) is a *separate diagnostic* for the feature-importance figure only. Negative mean R² is largely a **metric artifact**: low-degradation cells (A6 std 0.028, B5/B6 std ~0.018) have tiny `Var(y_test)`, so R² goes very negative even at low RMSE.

2. **Hardcoded Ns steps.** Pipeline assumes EIS at Ns={1,6} (37/33 freqs, 140 features) and capacity at Ns=8. Verified other datasets differ: GEIS-HC-RT = 33+33=132 feats, PEIS-sparseEIS = single Ns=5 (29 freqs), Na_NoEIS has no EIS. The hardcoded freq-count slicing in the experiment silently mis-slices on GEIS.

Outcome: a Zhang-faithful GPR run reporting honest metrics, plus a feature pipeline where Ns states / freq range / capacity step are injectable kwargs and feature layout flows downstream automatically. **Decisions made:** preserve originals via a backup git branch, then edit in place; supply dataset structure via plain function kwargs (no registry).

## Step 0 — Backup (before any edits)
- Commit current working state to a new branch `backup/pre-gpr-refactor`, then return to `develop`. This is the safety net for the in-place edits below. (`.gitignore`/`.claude/` etc. already modified — stash or include as-is in the backup commit.)

## Part A — Zhang-faithful GPR (edit in place)

Rewrite `scripts/algorithms/gpr.py`:
- Add `train_capacity_gpr(X_train, y_train, gpr_params=None)`:
  - Kernel `ConstantKernel() * RBF(length_scale=1.0) + WhiteKernel(noise_level=0.1, noise_level_bounds=(1e-5,1e1))` — **isotropic** single length scale, **learned** noise (Zhang `covSEiso` + `likGauss`).
  - `GaussianProcessRegressor(alpha=1e-10, normalize_y=True, n_restarts_optimizer=5+)`, fit on **full** X_train (no subsample — ~2500 pts/fold is sub-second). Reuse `_standardize_fit/_apply`.
- Add `train_ard_diagnostic(X_train, y_train)`: ARD model (`RBF(length_scale=ones(n_feat))`) used **only** to extract frequency weights; keep existing `ard_frequency_weights`.
- Keep `predict` returning `(mean, std)`. Retain the old `train_capacity_gpr_fast` as a clearly-commented deprecated path if cheap, else drop (backup branch has it).

Rewrite `experiments/gpr/gpr_loso.py`:
- Predict with isotropic model; run ARD diagnostic separately for weights.
- **Metrics:** keep per-fold RMSE/MAE; replace mean-of-per-fold-R² with (a) **pooled R²** over concatenated predictions vs global mean and (b) a **per-cell RMSE table**. Add a pooled parity plot (pred vs actual SOH).
- Build Re/Im ARD weight groupings from the **layout metadata** returned by the pipeline (Part B), not hardcoded 37/33.

## Part B — Injectable-Ns pipeline (plain kwargs)

Defaults preserve current behavior so nothing else breaks:
- `scripts/feature_engineering/state_vector.py`: add `freq_range` kwarg (default `(0.2,20000)`); already takes `ns_states`. **Return feature-layout metadata** alongside `(S, sample_ids)` — dict `{ns: {'freqs':[...], 'n':k}, 'order':[...]}` mapping columns -> (Ns, freq, Re/Im).
- `scripts/feature_engineering/main.py` `build_model_input`: accept `ns_states`, `freq_range`, `capacity_ns` (default 8) — replace hardcoded `df['Ns']==8`; pass through; return `(X, y, layout)`.
- `experiments/gpr/gpr_loso.py`: pass dataset's `ns_states`/`freq_range`/`capacity_ns` explicitly into `build_model_input`, and use returned `layout` for generic ARD slicing. Document example kwargs for PEIS-HC-RT {1,6}, GEIS-HC-RT {1,6}, PEIS-sparseEIS {5}.

## Verification
- Run `experiments/gpr/gpr_loso.py` on PEIS-HC-RT under `eis-ml-conda`; confirm it completes, prints per-cell RMSE + pooled R², and high-variance cells (A5/A7) fit well.
- Smoke-test `build_model_input` on GEIS-HC-RT (expect 132 feats) and PEIS-sparseEIS (single Ns, 58 feats) — no shape errors, correct `layout`.
- Confirm `git branch` shows `backup/pre-gpr-refactor` holding the pre-refactor code.
