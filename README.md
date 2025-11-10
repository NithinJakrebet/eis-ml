# EIS-ML: Battery Capacity Prediction using Impedance Spectroscopy

Machine learning framework for predicting battery degradation from Electrochemical Impedance Spectroscopy (EIS) measurements.

## Quick Start

### Environment Setup

Create and activate a conda environment with Python 3.12:

```bash
conda create -n eis-ml-conda python=3.12
conda activate eis-ml-conda
pip install -r requirements.txt
```

### Data Setup

 [[Download the dataset](https://drive.google.com/file/d/1oxKcQ_CtknQq9hlbonAoCJDUV-l3i-30/view?usp=share_link)] and extract into the `data/` directory.

## Workflow

### Notebooks vs Scripts

- **Notebooks** (`notebooks/`): For experimentation, prototyping, and interactive analysis. Not tracked in Git to avoid merge conflicts.
- **Scripts** (`scripts/`): Production-ready, reusable code. Tracked in Git for version control and team collaboration.

### Configuration-Driven Experiments

Model parameters are defined in YAML files (`configs/`) for easy modification without changing code.

Example workflow:
1. Edit parameters in `configs/gpr.yaml`
2. Run experiment notebook (e.g., `GPR_LOSO.ipynb`)
3. View results in MLflow UI
4. Commit config changes and result summaries to Git

## Experiment Tracking with MLflow

MLflow tracks experiments locally without requiring a shared server.

### Starting MLflow UI

```bash
conda activate eis-ml-conda
mlflow ui
```

Open http://127.0.0.1:5000 in your browser to view:
- Experiment runs with parameters and metrics
- Model performance plots
- ARD kernel analysis
- Cross-validation results

### Sharing Results

Since MLflow database (`mlruns/`) is not tracked in Git, share experiments by:
1. Saving key metrics/plots to `results/summaries/` (JSON/CSV)
2. Committing these lightweight files to Git
3. Team members can reproduce experiments using same configs

## Git Version Control

Files tracked in this repository:
- Production code (`scripts/`)
- Experiment configurations (`configs/`)
- Result summaries (`results/summaries/`)
- Documentation (`README.md`, `requirements.txt`)

Files excluded (see `.gitignore`):
- Raw data (`data/`)
- Trained models (`models/`)
- MLflow artifacts (`mlruns/`)
- Notebooks (`notebooks/`)
- Plot files (`results/plots/`)
- Virtual environments

This keeps the repository focused on code and configurations while avoiding large binary files.