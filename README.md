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
