# Graph Report - .  (2026-05-26)

## Corpus Check
- Corpus is ~35,016 words - fits in a single context window. You may not need a graph.

## Summary
- 140 nodes · 176 edges · 22 communities (18 shown, 4 thin omitted)
- Extraction: 85% EXTRACTED · 15% INFERRED · 0% AMBIGUOUS · INFERRED: 26 edges (avg confidence: 0.86)
- Token cost: 75,614 input · 25,204 output

## Community Hubs (Navigation)
- [[_COMMUNITY_SOH Pipeline & Issues|SOH Pipeline & Issues]]
- [[_COMMUNITY_Battery Datasets & States|Battery Datasets & States]]
- [[_COMMUNITY_Plotting Utilities|Plotting Utilities]]
- [[_COMMUNITY_Literature & Tracking|Literature & Tracking]]
- [[_COMMUNITY_ARD Weights Figure|ARD Weights Figure]]
- [[_COMMUNITY_Data Loading & Splits|Data Loading & Splits]]
- [[_COMMUNITY_GPR Predictions Figure|GPR Predictions Figure]]
- [[_COMMUNITY_GPR Training Code|GPR Training Code]]
- [[_COMMUNITY_Graphify Skill|Graphify Skill]]
- [[_COMMUNITY_State Vector Builder|State Vector Builder]]
- [[_COMMUNITY_Claude Hooks Config|Claude Hooks Config]]
- [[_COMMUNITY_RMSE Metric Rationale|RMSE Metric Rationale]]
- [[_COMMUNITY_Negative SOH Issue|Negative SOH Issue]]
- [[_COMMUNITY_Per-Cycle Fade Target|Per-Cycle Fade Target]]

## God Nodes (most connected - your core abstractions)
1. `battery-ml agent` - 11 edges
2. `GPR All Predictions Combined (State 6)` - 9 edges
3. `EIS-ML README` - 8 edges
4. `140-feature state vector` - 8 edges
5. `EIS-ML Project Context` - 7 edges
6. `ARD Weights Across Folds (State 6)` - 7 edges
7. `test_train_split()` - 6 edges
8. `load_single_channel()` - 6 edges
9. `graphify pipeline` - 6 edges
10. `GPR (Gaussian Process Regression) model` - 6 edges

## Surprising Connections (you probably didn't know these)
- `battery-ml agent` --references--> `Target leakage prohibition`  [EXTRACTED]
  .github/agents/battery-ml.agent.md → docs/CONTEXT.md
- `140-feature state vector` --shares_data_with--> `PEIS Ns=1 frequency grid (37 freqs)`  [EXTRACTED]
  docs/CONTEXT.md → README.md
- `140-feature state vector` --shares_data_with--> `PEIS Ns=6 frequency grid (33 freqs)`  [EXTRACTED]
  docs/CONTEXT.md → README.md
- `battery-ml agent` --references--> `EIS-ML Project Context`  [EXTRACTED]
  .github/agents/battery-ml.agent.md → docs/CONTEXT.md
- `battery-ml agent` --references--> `Cell B5 outlier (open issue)`  [EXTRACTED]
  .github/agents/battery-ml.agent.md → docs/CONTEXT.md

## Hyperedges (group relationships)
- **140-feature state vector layout (Ns=1 Re/Im + Ns=6 Re/Im)** — docs_context_state_vector_140, docs_context_ns_1, docs_context_ns_6, readme_freq_grid_ns1, readme_freq_grid_ns6 [EXTRACTED 1.00]
- **LOSO evaluation pipeline (loader → state vector → model → LOSO)** — docs_context_load_single_channel, docs_context_build_state_vector, docs_context_build_model_input, docs_context_loso, docs_context_evaluate_model [EXTRACTED 0.95]
- **ARD-based salient frequency identification (Zhang + this project)** — docs_context_ard, docs_context_gpr_model, docs_context_gpr_salient_freqs, docs_context_zhang_salient_freqs, docs_context_zhang_2020 [EXTRACTED 1.00]

## Communities (22 total, 4 thin omitted)

### Community 0 - "SOH Pipeline & Issues"
Cohesion: 0.16
Nodes (17): Cell B5 outlier (open issue), bin_and_split (capacity-binned diagnostic), build_model_input, build_state_vector, Channel identity invariant, Electrochemical Impedance Spectroscopy (EIS), experiments/gpr/gpr_loso.py (full 14-cell LOSO sweep), load_and_prepare_data (+9 more)

### Community 1 - "Battery Datasets & States"
Cohesion: 0.17
Nodes (16): Action vector, GEIS-HC-RT dataset, Na_NoEIS dataset, Ns=1 state (full discharge), Ns=3 state (charge current; sparseEIS label source), Ns=5 state (sparseEIS-only), Ns=6 state (full charge), Ns=8 state (CC discharge, capacity label source) (+8 more)

### Community 2 - "Plotting Utilities"
Cohesion: 0.15
Nodes (11): array, capacity_vs_cycle(), degradation(), model_predictions(), nyquist(), DataFrame, Plots model predictions:       - Predicted vs. True scatter (with optional 95% C, Plot residual analysis with optional metrics display.          Args:         res (+3 more)

### Community 3 - "Literature & Tracking"
Cohesion: 0.18
Nodes (12): EIS-ML Project Context, ARD (Automatic Relevance Determination), Gasper et al. 2022, GPR (Gaussian Process Regression) model, GPR ARD salient frequencies (4.8 Hz, 13.3 Hz), Intuitive Tutorial to GPR, Jones (XGB action-vector paper), Messing et al. 2021 (+4 more)

### Community 4 - "ARD Weights Figure"
Cohesion: 0.21
Nodes (13): Automatic Relevance Determination (ARD), ARD Weights, Cross-Validation Folds, Electrochemical Impedance Spectroscopy (EIS), Feature Importance Analysis, Frequency (Hz), Gaussian Process Regression (GPR), High Frequency Peak (~5-10 kHz) in Re(Z) (+5 more)

### Community 5 - "Data Loading & Splits"
Cohesion: 0.45
Nodes (7): load_single_channel(), load_and_prepare_data(), normalize_features(), bin_and_split(), loso(), temporal_split(), test_train_split()

### Community 6 - "GPR Predictions Figure"
Cohesion: 0.24
Nodes (11): Average R^2 = -0.328, Average RMSE = 0.0749, Capacity Axis (0 to ~1.0, normalized), GPR All Predictions Combined (State 6), GPR Model, 8-Fold LOSO Cross-Validation, Outlier Cells (B5 R^2=-4.013, B6 R^2=-10.022), Perfect Prediction Reference Line (+3 more)

### Community 7 - "GPR Training Code"
Cohesion: 0.29
Nodes (9): ard_frequency_weights(), predict_fast(), Predict with trained GPR model          Returns:         mean: Predicted capacit, Extract ARD importance weights from trained model          Returns exp(-length_s, Fast two-stage GPR training          Args:         X_train, y_train: Training da, _standardize_apply(), _standardize_fit(), _stratified_subsample() (+1 more)

### Community 8 - "Graphify Skill"
Cohesion: 0.20
Nodes (10): graphify skill, AST structural extraction, Community detection (Louvain), Graphify extraction subagent, God nodes, graphify pipeline, graphify query, Semantic LLM extraction (+2 more)

### Community 9 - "State Vector Builder"
Cohesion: 0.67
Nodes (3): build_model_input(), build_state_vector(), DataFrame

## Knowledge Gaps
- **34 isolated node(s):** `PreToolUse`, `array`, `DataFrame`, `graphify skill`, `Graphify extraction subagent` (+29 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **4 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `battery-ml agent` connect `SOH Pipeline & Issues` to `Battery Datasets & States`, `Literature & Tracking`?**
  _High betweenness centrality (0.060) - this node is a cross-community bridge._
- **Why does `140-feature state vector` connect `SOH Pipeline & Issues` to `Battery Datasets & States`?**
  _High betweenness centrality (0.037) - this node is a cross-community bridge._
- **Why does `EIS-ML Project Context` connect `Literature & Tracking` to `SOH Pipeline & Issues`?**
  _High betweenness centrality (0.024) - this node is a cross-community bridge._
- **What connects `PreToolUse`, `array`, `Plots model predictions:       - Predicted vs. True scatter (with optional 95% C` to the rest of the system?**
  _42 weakly-connected nodes found - possible documentation gaps or missing edges._