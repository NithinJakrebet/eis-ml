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
You can do this:
   ```bash
   cd eis-ml
   ```
or you can just open the newly created eis-ml folder from vscode, I would receommend this option.

### Setting Up a Virtual Environment and Installing Dependencies

It is recommended to use a virtual environment to manage the Python dependencies.

1. **Create a virtual environment** (requires Python 3.x):
   ```bash
   python3 -m venv venv
   ```
2. **Activate the virtual environment:**
   - **On Windows:**
     ```powershell
     Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass

     .\venv\Scripts\Activate
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
  Try to run the cell. Then select Python environment and then select the (venv)Python kernel.
  If that does not show up run the following.
  
  ```bash
  jupyter kernelspec list
  ```

   You should see the kernel:
  
    ```bash
   python3          <path>/eis-ml/venv/share/jupyter/kernels/python3
   ```
    Run the following
    ```bash
     pip uninstall jupyter jupyterlab notebook ipykernel
       pip install jupyter
    ```
   ```bash
   python -m ipykernel install --user --name=venv --display-name "Python (venv)"
    ```   

## Directory Structure

Below is an overview of the repository structure and a brief explanation of each folder:

```
eis-ml/
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
│                        # Write reusable functions here.
│
├── requirements.txt     # List of Python dependencies.
└── README.md            # Project documentation.
```

## Workflow Overview

### **1. Data Preparation**
- Preprocess the dataset and analyze feature correlations to select the most relevant ones for training.
- Use Jupyter notebooks in the `notebooks/` directory (e.g., `exploratory_analysis.ipynb`) to visualize and explore data trends.

### **2. Model Training**
- Train the model using either the appropriate notebook or script.
- Trained models will be stored in the `models/` directory as `.pkl` files using `joblib`.

### **3. Model Evaluation**
- Evaluate trained models using the evaluation script (`scripts/evaluate.py`).
- Performance metrics, visualizations, and logs will be saved in the `results/` directory.


## Branching Strategy

To maintain a clean and organized repository, follow this naming convention for feature branches:

- **Single developer working on a feature:**  
  ```
  <feature>
  ```
  Example: `preprocessing`

- **Multiple developers working on the same feature:**  
  ```
  <Name>-<feature>
  ```
  Example:  
  ```
  Nathan-preprocessing
  Dibo-preprocessing
  ```

This structure ensures clarity in collaboration and version control.  
