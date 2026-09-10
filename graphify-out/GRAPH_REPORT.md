# Graph Report - eis-ml  (2026-09-10)

## Corpus Check
- 27 files · ~19,275 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 507 nodes · 731 edges · 42 communities (31 shown, 11 thin omitted)
- Extraction: 95% EXTRACTED · 5% INFERRED · 0% AMBIGUOUS · INFERRED: 40 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Graph Freshness
- Built from commit: `2201644f`
- Run `git rev-parse HEAD` and compare to check if the graph is stale.
- Run `graphify update .` after code changes (no API cost).

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
- [[_COMMUNITY_Community 22|Community 22]]
- [[_COMMUNITY_Community 23|Community 23]]
- [[_COMMUNITY_Community 24|Community 24]]
- [[_COMMUNITY_Community 25|Community 25]]
- [[_COMMUNITY_Community 26|Community 26]]
- [[_COMMUNITY_Community 27|Community 27]]
- [[_COMMUNITY_Community 28|Community 28]]
- [[_COMMUNITY_Community 29|Community 29]]
- [[_COMMUNITY_Community 30|Community 30]]
- [[_COMMUNITY_Community 31|Community 31]]
- [[_COMMUNITY_Community 32|Community 32]]
- [[_COMMUNITY_Community 33|Community 33]]
- [[_COMMUNITY_Community 34|Community 34]]
- [[_COMMUNITY_Community 35|Community 35]]
- [[_COMMUNITY_Community 36|Community 36]]
- [[_COMMUNITY_Community 37|Community 37]]
- [[_COMMUNITY_Community 38|Community 38]]
- [[_COMMUNITY_Community 39|Community 39]]
- [[_COMMUNITY_Community 40|Community 40]]
- [[_COMMUNITY_Community 41|Community 41]]

## God Nodes (most connected - your core abstractions)
1. `DatasetSpec` - 26 edges
2. `loso_folds()` - 18 edges
3. `build_dataset()` - 17 edges
4. `What You Must Do When Invoked` - 16 edges
5. `eis_features()` - 15 edges
6. `/graphify` - 15 edges
7. `get_dataset()` - 14 edges
8. `EIS-ML: Battery State of Health from Impedance Spectroscopy` - 14 edges
9. `capacity_labels()` - 13 edges
10. `temporal_split()` - 13 edges

## Surprising Connections (you probably didn't know these)
- `DatasetSpec` --uses--> `DatasetSpec`  [INFERRED]
  features.py → datasets.py
- `Series` --uses--> `DatasetSpec`  [INFERRED]
  features.py → datasets.py
- `DatasetSpec` --uses--> `DatasetSpec`  [INFERRED]
  eis_ml/features.py → datasets.py
- `DatasetSpec` --uses--> `DatasetSpec`  [INFERRED]
  eis_ml/data.py → datasets.py
- `parity()` --calls--> `pooled()`  [EXTRACTED]
  plots.py → metrics.py

## Hyperedges (group relationships)
- **140-feature state vector layout (Ns=1 Re/Im + Ns=6 Re/Im)** — docs_context_state_vector_140, docs_context_ns_1, docs_context_ns_6, readme_freq_grid_ns1, readme_freq_grid_ns6 [EXTRACTED 1.00]
- **LOSO evaluation pipeline (loader → state vector → model → LOSO)** — docs_context_load_single_channel, docs_context_build_state_vector, docs_context_build_model_input, docs_context_loso, docs_context_evaluate_model [EXTRACTED 0.95]
- **ARD-based salient frequency identification (Zhang + this project)** — docs_context_ard, docs_context_gpr_model, docs_context_gpr_salient_freqs, docs_context_zhang_salient_freqs, docs_context_zhang_2020 [EXTRACTED 1.00]

## Communities (42 total, 11 thin omitted)

### Community 0 - "SOH Pipeline & Issues"
Cohesion: 0.06
Nodes (45): EIS-ML Project Context, Action vector, ARD (Automatic Relevance Determination), Cell B5 outlier (open issue), bin_and_split (capacity-binned diagnostic), build_model_input, build_state_vector, Channel identity invariant (+37 more)

### Community 1 - "Battery Datasets & States"
Cohesion: 0.05
Nodes (72): bool, DataFrame, int, str, DatasetSpec, cell_path(), list_cells(), load_cell() (+64 more)

### Community 2 - "Plotting Utilities"
Cohesion: 0.15
Nodes (11): array, capacity_vs_cycle(), degradation(), model_predictions(), nyquist(), DataFrame, Plots model predictions:       - Predicted vs. True scatter (with optional 95% C, Plot residual analysis with optional metrics display.          Args:         res (+3 more)

### Community 3 - "Literature & Tracking"
Cohesion: 0.04
Nodes (47): graphify skill, AST structural extraction, Community detection (Louvain), Graphify extraction subagent, God nodes, graphify pipeline, graphify query, Semantic LLM extraction (+39 more)

### Community 4 - "ARD Weights Figure"
Cohesion: 0.21
Nodes (13): Automatic Relevance Determination (ARD), ARD Weights, Cross-Validation Folds, Electrochemical Impedance Spectroscopy (EIS), Feature Importance Analysis, Frequency (Hz), Gaussian Process Regression (GPR), High Frequency Peak (~5-10 kHz) in Re(Z) (+5 more)

### Community 5 - "Data Loading & Splits"
Cohesion: 0.26
Nodes (10): load_single_channel(), load_and_prepare_data(), normalize_features(), bin_and_split(), loso(), temporal_split(), test_train_split(), build_model_input() (+2 more)

### Community 6 - "GPR Predictions Figure"
Cohesion: 0.24
Nodes (11): Average R^2 = -0.328, Average RMSE = 0.0749, Capacity Axis (0 to ~1.0, normalized), GPR All Predictions Combined (State 6), GPR Model, 8-Fold LOSO Cross-Validation, Outlier Cells (B5 R^2=-4.013, B6 R^2=-10.022), Perfect Prediction Reference Line (+3 more)

### Community 7 - "GPR Training Code"
Cohesion: 0.13
Nodes (23): ard_frequency_weights(), ard_weights(), fit(), fit_ard(), _fit_gp(), _kernel(), predict(), predict_fast() (+15 more)

### Community 8 - "Graphify Skill"
Cohesion: 0.22
Nodes (8): feature_importance(), fit(), predict(), ndarray, XGBoost ensemble for SOH, following Jones et al. (2022).  ``n_models`` regressor, Train an ensemble; extra keyword arguments override ``DEFAULT_PARAMS``., Ensemble mean and standard deviation., Gain-based importance averaged over ensemble members.

### Community 9 - "State Vector Builder"
Cohesion: 0.06
Nodes (36): code:bash (mkdir -p graphify-out), code:bash ($(cat graphify-out/.graphify_python) -c "), code:bash (LOCAL_PATH=$(graphify clone <github-url> [--branch <branch>]), code:bash (graphify export obsidian), code:bash (graphify export html  # auto-aggregates to community view if), code:bash (graphify export wiki), code:bash (graphify export neo4j), code:bash (graphify export neo4j --push bolt://localhost:7687 --user ne) (+28 more)

### Community 22 - "Community 22"
Cohesion: 0.23
Nodes (21): ard_weights(), calibration(), capacity_vs_cycle(), degradation(), nyquist(), parity(), DataFrame, int (+13 more)

### Community 23 - "Community 23"
Cohesion: 0.14
Nodes (17): channels(), loso_folds(), DataFrame, float, str, Train/test splits over a (channel, cycle)-indexed feature matrix.  Leave-one-cel, Yield ``(held_out_cell, train_mask, test_mask)`` for every cell.      ``cells``, Within each cell, train on the earliest ``train_frac`` of cycles.      Diagnosti (+9 more)

### Community 24 - "Community 24"
Cohesion: 0.11
Nodes (21): 1. Setup, 2. Data, 3. Run an experiment, 4. Using the package directly, 5. Protocols and labels, 6. Evaluation, 7. Workflow and git, code:bash (conda create -n eis-ml python=3.12 && conda activate eis-ml) (+13 more)

### Community 25 - "Community 25"
Cohesion: 0.14
Nodes (15): code:block1 (load_dataset(spec) -> df           long form, all cells, 'ch), Conventions, Data, EIS-ML Project Context, Evaluation, Goal, Historical bugs, do not regress, Jones et al., Nature Communications 13:4806 (2022) (+7 more)

### Community 26 - "Community 26"
Cohesion: 0.14
Nodes (13): Architecture, Backend — Python (`visualizer/api/`), code:block1 (eis-ml/visualizer/), File Structure, Frontend — React (`visualizer/client/`), Goal: EIS Data Visualization Dashboard, Known Data Quirks to Handle Gracefully, Layout (+5 more)

### Community 27 - "Community 27"
Cohesion: 0.15
Nodes (13): code:bash ($(cat graphify-out/.graphify_python) -c "), code:block11 ([Agent tool call 1: files 1-15, subagent_type="general-purpo), code:bash (PROJECT_ROOT=$(cat graphify-out/.graphify_root)), code:block13 (You are a graphify extraction subagent. Read the files liste), code:bash ($(cat graphify-out/.graphify_python) -c "), code:bash ($(cat graphify-out/.graphify_python) -c "), code:bash ($(cat graphify-out/.graphify_python) -c "), code:bash ($(cat graphify-out/.graphify_python) -c ") (+5 more)

### Community 28 - "Community 28"
Cohesion: 0.33
Nodes (14): by_soh_band(), coverage(), per_cell(), pooled(), DataFrame, float, str, Evaluation of LOSO predictions.  All functions take the "predictions" table writ (+6 more)

### Community 29 - "Community 29"
Cohesion: 0.22
Nodes (8): 1. GPR baseline fixed (Zhang-faithful), 2. Injectable-Ns feature pipeline, 3. XGBoost vs Jones et al. (2022), 4. Multi-step forecasting (Jones Fig 4, adapted), EIS-ML Progress Update — Lab Meeting (2026-06-01), In progress / next steps, Scientific takeaways, TL;DR

### Community 30 - "Community 30"
Cohesion: 0.25
Nodes (7): battery-ml, Hard rules — never violate, How to operate, Things to flag, not silently fix, tools: ['vscode', 'execute', 'read', 'agent', 'edit', 'search', 'web', 'todo'], What you do, When unsure

### Community 31 - "Community 31"
Cohesion: 0.25
Nodes (7): Avoid redundancy (already exists), Context, Methods vs Jones (for the writeup), Plan 02 — XGBoost evaluation + Jones (2022) comparison, Proposed work (pick scope via the question I will ask), Verification, Workflow

### Community 32 - "Community 32"
Cohesion: 0.29
Nodes (6): Context, Part A — Zhang-faithful GPR (edit in place), Part B — Injectable-Ns pipeline (plain kwargs), Plan: Zhang-faithful GPR + injectable-Ns pipeline, Step 0 — Backup (before any edits), Verification

### Community 33 - "Community 33"
Cohesion: 0.33
Nodes (5): Context, Out of scope (later plans), Plan 01 — Diagnostics: per-cell / per-SOH-band error + uncertainty calibration, Steps, Verification

### Community 34 - "Community 34"
Cohesion: 0.33
Nodes (5): GPR LOSO (Zhang-faithful) — Results Analysis, ML & Physics, Headline, ML perspective (parity plot), Next steps, Physics perspective (ARD relevance weights)

## Knowledge Gaps
- **155 isolated node(s):** `str`, `float`, `ndarray`, `ndarray`, `PreToolUse` (+150 more)
  These have ≤1 connection - possible missing edges or undocumented components.
- **11 thin communities (<3 nodes) omitted from report** — run `graphify query` to explore isolated nodes.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `What You Must Do When Invoked` connect `State Vector Builder` to `Community 27`, `Literature & Tracking`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Why does `/graphify` connect `Literature & Tracking` to `State Vector Builder`?**
  _High betweenness centrality (0.026) - this node is a cross-community bridge._
- **Are the 12 inferred relationships involving `DatasetSpec` (e.g. with `DataFrame` and `int`) actually correct?**
  _`DatasetSpec` has 12 INFERRED edges - model-reasoned connections that need verification._
- **What connects `Where does the error live along the degradation axis?`, `Fraction of residuals inside 1 and 2 predicted sigma (nominal 68% / 95%).`, `Where to find EIS spectra and capacity labels inside one protocol.      name:` to the rest of the system?**
  _209 weakly-connected nodes found - possible documentation gaps or missing edges._
- **Should `SOH Pipeline & Issues` be split into smaller, more focused modules?**
  _Cohesion score 0.0647342995169082 - nodes in this community are weakly interconnected._
- **Should `Battery Datasets & States` be split into smaller, more focused modules?**
  _Cohesion score 0.054363796650014694 - nodes in this community are weakly interconnected._
- **Should `Literature & Tracking` be split into smaller, more focused modules?**
  _Cohesion score 0.041666666666666664 - nodes in this community are weakly interconnected._