# EIS 

This repository contains a machine learning framework for analyzing battery performance using Electrochemical Impedance Spectroscopy (EIS) data. It includes data processing scripts, model training and evaluation code, and Jupyter notebooks for exploratory analysis.       

## Getting Started

### Clone the Repository

1. **Open Visual Studio Code.**
2. **Open the Integrated Terminal** by pressing <kbd>Ctrl</kbd>+<kbd>`</kbd> (or via the menu: *View > Terminal*).
3. **Clone the repository** by running:
   ```bash
   git clone https://github.com/NithinJakrebet/eis-ml.git
   ```
4. **Navigate into the project directory:**
   ```bash
   cd eis-ml
   ```

### Setting Up a Virtual Environment and Installing Dependencies

It is recommended to use a virtual environment to manage the Python dependencies.

1. **Create a virtual environment** (requires Python 3.x):
   ```bash
   python3 -m venv venv
   ```
2. **Activate the virtual environment:**
   - **On Windows:**
     ```bash
     venv\Scripts\activate
     ```
   - **On macOS and Linux:**
     ```bash
     source venv/bin/activate
     ```
3. **Upgrade pip (optional but recommended):**
   ```bash
   pip install --upgrade pip
   ```
4. **Install the required packages:**
   ```bash
   pip install -r requirements.txt
   ```

## Directory Structure

Below is an overview of the repository structure and a brief explanation of each folder:

```
your-repo/
│
├── data/                # Contains EIS data files.
│
├── models/              # Directory for saving and loading trained model files.
│
├── notebooks/           # Jupyter notebooks for exploratory analysis, model training, and evaluation.
│
├── results/             # Holds the outputs such as evaluation logs, plots, and performance metrics.
│
├── scripts/             # Python scripts for data preprocessing, model training, and evaluation.
│
├── requirements.txt     # List of Python dependencies.
└── README.md            # This file.
```

# Raw Data Information (Hardware Team)

## Li_lowC_lowT_25C (03/22/24)

* **Cells**: 8 (B1–B8)
* **Cycles**: ~150
* **Protocol**: Low C-rate
* **Temperature**: B1–B4 (Low T), B5–B8 (25 °C)
* **EIS**: Yes
* **Notes**: Baseline low-rate dataset for comparing temperature effects on Li-ion cycling.

## Li_highC_25C (04/03/24)

* **Cells**: 8 (A1–A8)
* **Cycles**: ~265
* **Protocol**: High C-rate
* **Temperature**: 25 °C
* **EIS**: Yes
* **Failures**: All channels except A3 and A6 reached end of life.

## Li_2C-3.75C_25C_precycled (06/10/24)

* **Cells**: 8 (A1–A8)
* **Chemistry**: Li-ion (Molicell 21700 P42A, NMC)
* **Protocol**: 2C charge / 3.75C discharge
* **EIS**: Every 10 cycles
* **Temperature**: 25 °C
* **Equipment**: BCS-815 (SN 1054), BT-Lab v1.79
* **Notes**: Precycled cells; occasional impedance spikes; suitable for EIS-based degradation modeling.

## Na_1.3C-3C_25C_noEIS_fastcycle (06/18/24)

* **Cells**: 2 (A1–A2)
* **Chemistry**: Na-ion 18650
* **Protocol**: 1.3C charge / 3C discharge until failure
* **EIS**: None (continuous cycling only)
* **Temperature**: 25 °C
* **Equipment**: BCS-815 (SN 1054), BT-Lab v1.79
* **Notes**: Uses `fastcyclewithoutEISnaion.mps`; fast-degradation Na-ion test; no impedance data for modeling (use for capacity-fade comparison only).


# EIS-ML Project Overview

## Development Approach
- **Structured code → Python scripts** (`scripts/` folder)
- **Exploration & EDA → Jupyter notebooks** (`notebooks/` folder)

## Data Structure

**Key Data Points:**
- EIS measurements at `Ns=1` or `Ns=6` (PEIS steps)
- Frequency range: 0.02Hz - 20kHz ( 69 frequencies)
- Current data from CC/CV steps for action vectors

## Machine Learning Models
1. **Gaussian Process Regression (GPR)** with ARD kernels
2. **XGBoost** (Gradient Boosted Decision Trees)

## Regression Task
Learn mapping: `Qn = f(sn, an)` where:
- `sn` = battery state (EIS impedance)
- `Qn` = resulting capacity
