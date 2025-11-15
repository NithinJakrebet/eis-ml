# EIS-ML: Battery Capacity Prediction Using Impedance Spectroscopy

This repository contains data processing tools, modeling code, and experiment scripts for predicting Li-ion battery capacity from Electrochemical Impedance Spectroscopy (EIS).
The project uses both PEIS and GEIS datasets collected under high-C charge/discharge regimes at room temperature.

All development and analysis are performed in Jupyter notebooks.
When an experiment is successful, it is exported as a standalone Python script for reproducibility.

---

# 1. Environment Setup

Create and activate the Conda environment:

```bash
conda create -n eis-ml python=3.12
conda activate eis-ml
pip install -r requirements.txt
```

---

# 2. Data Setup

Download the dataset and place it inside the `data/` directory:

**Download:**
[https://drive.google.com/file/d/1oxKcQ_CtknQq9hlbonAoCJDUV-l3i-30/view?usp=share_link](https://drive.google.com/file/d/1oxKcQ_CtknQq9hlbonAoCJDUV-l3i-30/view?usp=share_link)

Data should follow this structure:

```
data/
    PEIS-HC-RT/
    PEIS-HC-RT-sparseEIS/
    Na_NoEIS/
    GEIS-HC-RT/
```

No raw data files are tracked in Git.

---

# 3. Workflow

## 3.1 Notebooks and Scripts

**notebooks/**
Interactive exploration, EDA, feature design, and model development.
These files are not tracked in Git.

**scripts/**
Reusable code for feature engineering and data processing

**experiments/**
When a notebook produces a result worth preserving, export it:

Jupyter → File → Export As → Python Script

Place the script inside `experiments/` as a reproducible experiment.

## 3.2 Shared Library Code

Reusable utilities (data loading, preprocessing, feature alignment, plotting, model wrappers) live in:

```
scripts/
```

Experiment scripts import from these modules to avoid duplicated code.

# 4. Dataset Families and Experimental Protocols

All datasets were collected at approximately 25°C with high-C charge and discharge.

Below is a full description of the experimental families.

---

## 4.1 PEIS-HC-RT

Potentiostatic EIS • High C-rates • Room Temperature • EIS Every Cycle

**EIS Mode:** PEIS (voltage-driven, ~10 mV)
**Charge:** 2C CC + CV at 4.2 V
**Discharge:** 3.75C CC to 2.0 V
**EIS Steps:**

* Ns = 1 (before charge)
* Ns = 6 (after charge)

### Frequency Grids

**Ns = 1 (~37 points):**
0.254–10,000 Hz (log-spaced)

**Ns = 6 (~33 points):**
~1–10,000 Hz (log-spaced)

Exact frequency lists are included later in this README.

### Capacity Labels

* Extract from **Ns = 8** (constant-current discharge)
* Use the final or maximum value of `Capacity/mA.h`
* Validate that median `I/mA < 0` (discharge)

### Datasets

* **04-03-24_A1–A8 (SOH Project)** — BCS-815
* **03-07/03-08-24_B1–B6 (CB series)** — BCS-810

---

## 4.2 PEIS-HC-RT-sparseEIS

Potentiostatic EIS • High C-rates • Room Temperature • EIS Every 10 Cycles

Same charge/discharge regime as PEIS-HC-RT.
EIS is performed only when Ns = 5 appears in the cycle.

### EIS Step

* Ns = 5 only
* Keep only cycles that actually contain Ns = 5

### Capacity Labels

* Extract from **Ns = 3** (constant-current discharge)

### Dataset

* **06-10-24_A1–A8 (Patricio)** — BCS-815

---

## 4.3 GEIS-HC-RT

Galvanostatic EIS • High C-rates • Room Temperature • EIS Every Cycle

**EIS Mode:** GEIS (current-driven, ~100 mA)
**EIS Steps:**

* Ns = 1
* Ns = 6
  **Frequency Range:** ~1–10,000 Hz for both Ns states

### Capacity Labels

* Extract from **Ns = 8**

### Dataset

* **03-07-24_B7–B8 (CB series)** — BCS-810

---

# 5. Frequency Grids

These grids are used for interpolation, feature extraction, and model input construction.

## 5.1 PEIS Ns = 1 (37 frequencies)

```
0.254, 0.34, 0.456, 0.612, 0.822, 1.1, 1.48, 1.99,
2.66, 3.57, 4.8, 6.43, 8.64, 11.6, 15.5,
20.9, 28.0, 37.5, 50.3, 67.6, 90.6, 122.0, 163.0,
219.0, 294.0, 394.0, 529.0, 710.0, 952.0, 1280.0, 1710.0,
2300.0, 3090.0, 4140.0, 5560.0, 7450.0, 10000.0
```

## 5.2 PEIS Ns = 6 (33 frequencies)

```
0.999, 1.33, 1.78, 2.37, 3.16, 4.22, 5.62, 7.5, 10.0, 13.3,
17.8, 23.7, 31.6, 42.2, 56.2, 75.0, 102.0, 135.0, 178.0, 237.0,
316.0, 422.0, 564.0, 750.0, 1000.0, 1330.0, 1780.0, 2370.0,
3160.0, 4220.0, 5620.0, 7500.0, 10000.0
```

---

# 6. Modeling Guidelines

## 6.1 Feature Alignment

**PEIS-only models:**
Use full Ns=1 grid or interpolate both Ns=1 and Ns=6 to 0.254–10 kHz.

**PEIS + GEIS or cross-step comparisons:**
Use shared 1–10 kHz grid.

## 6.2 Capacity Label Rules

* PEIS-HC-RT and GEIS-HC-RT use **Ns = 8**
* PEIS-sparseEIS uses **Ns = 3**
* Always choose final or max capacity within that Ns step
* Confirm discharge using negative median current

## 6.3 Cycle Alignment

* Pair EIS and capacity from the **same cycle number**
* For sparseEIS, keep only cycles where Ns = 5 exists

## 6.4 Quality Control

* Drop EIS points with obvious glitches at frequency band edges
* Prefer Zre and -Zim instead of magnitude/phase unless needed
* Tag samples by cell for LOSO evaluation

---

# 7. Git Version Control

Tracked:

* `scripts/`
* `src/`
* `results/`
* `experiments/`
* Documentation files

Ignored:

* `data/`
* Jupyter notebooks
* Trained models
* MLflow artifacts (if generated)
* Large images or plots

This keeps the repository lightweight and focuses on reproducible code.