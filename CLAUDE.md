# eis-ml

Predict Li-ion battery state of health (SOH) from EIS spectra. Read `docs/CONTEXT.md`
before non-trivial work; it is the orientation document for this repo.

Quick facts:
- Modules sit at the repo root (`datasets.py`, `features.py`, `splits.py`, `algorithms/`, `metrics.py`, `plots.py`); `pip install -r requirements.txt`. Experiments: `experiments/`. Tests: `pytest`.
- Python env: conda `eis-ml-conda` (`/opt/anaconda3/envs/eis-ml-conda/bin/python`).
- Run a LOSO experiment: `python experiments/run_loso.py --dataset PEIS-HC-RT --model gpr --ns 6`.
- The EIS step (`Ns`) is injectable through `get_dataset(name, eis_ns=[...])` or `--ns`; never hardcode it.
- Results go to `results/<model>/<dataset>_ns<steps>/`; raw data, notebooks, models and plots are not tracked.

## graphify

This project has a knowledge graph at graphify-out/ with god nodes, community structure, and cross-file relationships.

Rules:
- For codebase questions, first run `graphify query "<question>"` when graphify-out/graph.json exists. Use `graphify path "<A>" "<B>"` for relationships and `graphify explain "<concept>"` for focused concepts. These return a scoped subgraph, usually much smaller than GRAPH_REPORT.md or raw grep output.
- If graphify-out/wiki/index.md exists, use it for broad navigation instead of raw source browsing.
- Read graphify-out/GRAPH_REPORT.md only for broad architecture review or when query/path/explain do not surface enough context.
- After modifying code, run `graphify update .` to keep the graph current (AST-only, no API cost).
