#!/usr/bin/env python
"""
batch_evaluation.py

This script loads multiple CSV datasets from different folders,
preprocesses them to build the state and action vectors (using the functions
from state_and_action_vector_preprocess.py), runs a pre-trained ensemble
of gradient boosting models on the combined feature set, evaluates the predictions,
and writes the evaluation metrics for each CSV file into a results CSV file.

Folders structure example:
    - path/to/3-6-24_A8_Channels/
          A1.csv, A2.csv, ..., A8.csv
    - path/to/3-22-24_A8_B8_Channels/
          A1.csv, ..., A8.csv, B1.csv, ..., B8.csv
    - path/to/04-03-24/
          A1.csv, A2.csv, ..., A8.csv

Usage:
    python batch_evaluation.py
"""

import os
import glob
import joblib
import numpy as np
import pandas as pd

# Import your preprocessing functions and evaluation function
from state_and_action_vector_preprocess import (
    build_eis_data_by_cycle,
    build_action_data_by_cycle,
    build_combined_dataset
)
from evaluate import evaluate_model
from load_data import load_data

# === CONFIGURATION ===
# List of folders to process (update these paths to your actual directories)
folders = [
    "data/03-06-24",
    "data/03-22-24",
    "data/04-03-24"
]

# Path to the pickled ensemble model
MODEL_PATH = "models/gb_ensemble.pkl"

# Output CSV file for evaluation results
OUTPUT_CSV = "results/model_evaluation_results.csv"

# === LOAD THE PRE-TRAINED MODEL ENSEMBLE ===
print("Loading ensemble models from", MODEL_PATH)
models = joblib.load(MODEL_PATH)  # models is expected to be a list of GB regressors

# === LOOP THROUGH CSV FILES in Each Folder ===
results = []

for folder in folders:
    # Use glob to find all CSV files in the folder
    csv_files = glob.glob(os.path.join(folder, "*.csv"))
    print(f"Processing folder: {folder} ({len(csv_files)} files found)")
    for csv_file in csv_files:
        try:
            # Load the CSV file into a DataFrame.
            df = pd.read_csv(csv_file)
            # Drop NaN values
            df = df.dropna()

            # Fix column name issue
            if "#NAME?" in df.columns:
                df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
                
            # drop the 0th cycle
            df = df.loc[(df['cycle number'] != 0)].copy()
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
            continue

        # Preprocess: build the EIS state and action vectors
        eis_dict = build_eis_data_by_cycle(df)
        action_dict = build_action_data_by_cycle(df)
        X, y = build_combined_dataset(
            df,
            eis_data_by_cycle=eis_dict,
            action_data_by_cycle=action_dict,
        )
        
        # If no cycles were processed, skip this file
        if X.shape[0] == 0:
            print(f"Skipping {csv_file}: no valid cycles found.")
            continue

        # Get predictions from the ensemble:
        preds_ensemble = [model.predict(X) for model in models]
        preds_ensemble = np.array(preds_ensemble)  # shape: (num_models, num_samples)
        y_pred_mean = np.mean(preds_ensemble, axis=0)
        y_pred_std = np.std(preds_ensemble, axis=0)

        # Evaluate predictions using your evaluate_model function
        rmse, r2, mse, mae = evaluate_model(y, y_pred_mean)

        # Collect results; you can add more columns as desired.
        result = {
            "folder": os.path.basename(folder),
            "file": os.path.basename(csv_file),
            "num_cycles": X.shape[0],
            "rmse": rmse,
            "r2": r2,
            "mse": mse,
            "mae": mae,
            "mean_prediction": np.mean(y_pred_mean),
            "mean_prediction_std": np.mean(y_pred_std)
        }
        results.append(result)
        print(f"Processed {os.path.basename(csv_file)}: {result}")

# === SAVE THE RESULTS TO A CSV FILE ===
results_df = pd.DataFrame(results)
results_df.to_csv(OUTPUT_CSV, index=False)
print("Evaluation results saved to", OUTPUT_CSV)
