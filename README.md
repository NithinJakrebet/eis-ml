# Battery ML Creation Model from EIS Data

This repository contains a machine learning framework for analyzing battery performance using Electrochemical Impedance Spectroscopy (EIS) data. It includes data processing scripts, model training and evaluation code, and Jupyter notebooks for exploratory analysis.

### exiting weird terminal venv
exec zsh    
source venv/bin/activate            

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

### Running the Project

- **Jupyter Notebooks:**  
  Open the notebook files in the `notebooks/` directory directly in VSCode or launch Jupyter Notebook from the terminal:
  ```bash
  jupyter notebook
  ```
- **Python Scripts:**  
  Ensure your virtual environment is activated, then run any script from the `scripts/` directory:
  ```bash
  python scripts/your_script.py
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

# Raw Data Information from Hardware Team

## 3/22/24 - A8, B8 Channels from Selva
- **Number of Cycles**: ~150
- **Details**:
  - Channels: 
    - **B1 to B4**: Low C-Rate, Low Temperature (?? °C)
    - **B5 to B8**: Low C-Rate, Room Temperature (25 °C)

## 04/03/24
- **Number of Cycles**: ~265
- **Details**:
  - Channels: A1 to A8
  - High C-Rate, Room Temperature (25 °C)
  - **Failures**:
    - All channels except A3 and A6 have achieved failure

# EIS-ML Project Overview

## Development Approach
- **Structured code → Python scripts** (`scripts/` folder)
- **Exploration & EDA → Jupyter notebooks** (`notebooks/` folder)

## Data Structure
**Input Features:**
- **State Vector**: Real and imaginary impedance components [Zre(ω1), Zim(ω1), ..., Zre(ωn), Zim(ωn)] from EIS measurements
- **Action Vector**: Charge/discharge current profiles during battery cycling
- **Target**: Battery discharge capacity (mAh)

**Key Data Points:**
- EIS measurements at `Ns=1` or `Ns=6` (PEIS steps)
- Frequency range: 0.02Hz - 20kHz (~60 frequencies)
- Current data from CC/CV steps for action vectors

## Machine Learning Models
1. **Gaussian Process Regression (GPR)** with ARD kernels
2. **XGBoost** (Gradient Boosted Decision Trees)

## Regression Task
Learn mapping: `Qn = f(sn, an)` where:
- `sn` = battery state (EIS impedance)
- `an` = action (charge/discharge protocol)  
- `Qn` = resulting capacity
