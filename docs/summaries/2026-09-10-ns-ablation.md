# EIS-state (Ns) ablation (2026-09-10)

**Branch:** `ns-ablation`. **Code:** `experiments/ns_ablation.py` (runs `run_loso` for Ns=1, Ns=6
and Ns=1+6 per model and collects the summaries). Per-run artefacts under
`results/ablation/ns/<model>/PEIS-HC-RT_ns<steps>/`; table and chart in `results/ablation/ns/`.

## PEIS-HC-RT, 14-cell LOSO, pooled RMSE (SOH)

| EIS state | features | GPR | XGB | XGB RMSE below SOH 0.4 |
|---|---|---|---|---|
| Ns 1 (discharged rest) | 74 | 0.145 | 0.052 | 0.052 |
| Ns 6 (charged rest) | 66 | 0.144 | 0.044 | 0.049 |
| Ns 1 + 6 | 140 | 0.132 | 0.042 | 0.049 |

## Reading
- **The charged-state sweep carries most of the usable signal for XGB**: Ns 6 alone is within
  0.002 RMSE of both states together, while Ns 1 alone is 25% worse. This matches the ARD
  diagnostic (73% of relevance on Ns 6) and Zhang's finding that the rested, fully charged
  state is the most predictive one.
- For GPR the two states are equivalent alone (0.145 vs 0.144) and only their combination
  helps (0.132). The GPR numbers are dominated by the low-SOH mean-reversion problem, not by
  which state is used; see the `gpr-mean-function` branch, where a linear mean function brings
  GPR to 0.050 with all 140 features.
- Practical implication: if measurement time matters, one sweep at the charged rest (Ns 6)
  is enough for XGB-level accuracy; the discharged sweep adds little.
