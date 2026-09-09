# Legacy results (pre-refactor, June 2026 and earlier)

Produced by the old `scripts/` pipeline and are kept for reference only.

- `gpr/gpr_8_fold_cv_results.*`, `gpr/ard_weights_across_folds.csv`, `gpr/state_6/`,
  `gpr/gpr-select-freqs_*`: the **old two-stage ARD GPR** (fixed near-zero noise,
  hyperparameters from a 300-sample subset). Not the Zhang-faithful isotropic model.
- `gpr/outputs/`: first Zhang-faithful isotropic run (pooled R2 0.792, RMSE 0.146) and its ARD weights.
- `xgb/xgb_8_fold_cv_results.*`: 14-cell XGB LOSO from the notebook era.
- `xgb/outputs/xgb_multistep_*`: horizon-as-action forecasting curve from the notebook.

Current results live in `results/<model>/<dataset>_ns<steps>/` and are produced by
`experiments/run_loso.py` and `experiments/xgb_multistep.py`.
