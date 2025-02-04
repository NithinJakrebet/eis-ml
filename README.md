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
   git clone https://github.com/your-username/your-repo.git
   ```
   Replace `https://github.com/your-username/your-repo.git` with your repository's URL.
4. **Navigate into the project directory:**
   ```bash
   cd your-repo
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
├── data/                
│   ├── raw/             # Contains raw, unprocessed EIS data files.
│   └── processed/       # Contains cleaned and preprocessed data.
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

### Workflow Overview

1. **Data Preparation:**
   - **Place raw data** in the `data/raw/` directory.
   - Run the data preprocessing script from the `scripts/` folder (e.g., `scripts/data_preprocessing.py`) to clean the data and save it into `data/processed/`.

2. **Exploratory Analysis:**
   - Use the notebooks in the `notebooks/` directory (for example, `exploratory_analysis.ipynb`) to explore and visualize the data.

3. **Model Training:**
   - Train your model by running the appropriate notebook (e.g., `model_training.ipynb`) or executing the training script in `scripts/train.py`.
   - The trained models will be saved in the `models/` directory.

4. **Model Evaluation:**
   - Evaluate your trained model using either the evaluation notebook (e.g., `evaluation.ipynb`) or by running the evaluation script in `scripts/evaluate.py`.
   - Performance metrics, plots, and logs will be stored in the `results/` directory.

