"""
Data pipeline for EIS-ML project
Handles data loading, preprocessing, and preparation for model training
"""

import pandas as pd
import numpy as np
from feature_engineering import build_model_input, drop_last_cycle
import config
import os

def load_data(subfolder: str, filename: str, data_dir: str = "../data") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, subfolder, filename))
    df = df.dropna()

    # Fix column name issue
    if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
        
    # drop the 0th cycle
    df = df.loc[(df['cycle number'] != 0)].copy()

    return df


def load_and_prepare_data(data_folder=None):
    """
    Load and prepare training and testing data.
    
    Args:
        data_folder (str): Folder containing the CSV files. If None, uses DEFAULT_DATA_FOLDER
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    if data_folder is None:
        data_folder = config.DEFAULT_DATA_FOLDER
    
    # Load data by channel
    data_by_channel = {}
    for channel in config.CHANNELS:
        try:
            df = load_data(data_folder, f"{channel}.csv")
            df = drop_last_cycle(df)
            data_by_channel[channel] = df
            print(f"Loaded {channel}: {df.shape[0]} rows")
        except FileNotFoundError:
            print(f"Warning: {channel}.csv not found in {data_folder}")
            continue
    
    # Separate training and testing data
    train_dfs = [data_by_channel[ch] for ch in config.TRAIN_CHANNELS if ch in data_by_channel]
    test_dfs = [data_by_channel[ch] for ch in config.TEST_CHANNELS if ch in data_by_channel]
    
    if not train_dfs:
        raise ValueError("No training data found!")
    if not test_dfs:
        raise ValueError("No testing data found!")
    
    # Concatenate DataFrames
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    print(f"Training data: {df_train.shape[0]} rows")
    print(f"Testing data: {df_test.shape[0]} rows")
    
    # Build feature matrices
    X_train, cycles_train = build_model_input(df_train)
    X_test, cycles_test = build_model_input(df_test)
    
    # Build target vectors
    last_discharge_train = df_train.groupby('cycle number')['Q discharge/mA.h'].last()
    y_train = last_discharge_train.loc[cycles_train].values
    
    last_discharge_test = df_test.groupby('cycle number')['Q discharge/mA.h'].last()
    y_test = last_discharge_test.loc[cycles_test].values
    
    print(f"X_train shape: {X_train.shape}, y_train shape: {y_train.shape}")
    print(f"X_test shape: {X_test.shape}, y_test shape: {y_test.shape}")
    
    return X_train, X_test, y_train, y_test

def get_data_summary(data_folder=None):
    """
    Get a summary of data availability across channels and folders.
    
    Args:
        data_folder (str): Folder to analyze. If None, uses DEFAULT_DATA_FOLDER
        
    Returns:
        dict: Summary of data availability
    """
    if data_folder is None:
        data_folder = config.DEFAULT_DATA_FOLDER
    
    summary = {}
    for channel in config.CHANNELS:
        try:
            df = load_data(data_folder, f"{channel}.csv")
            summary[channel] = {
                'rows': df.shape[0],
                'cycles': df['cycle number'].nunique() if 'cycle number' in df.columns else 0,
                'available': True
            }
        except FileNotFoundError:
            summary[channel] = {'available': False}
    
    return summary
