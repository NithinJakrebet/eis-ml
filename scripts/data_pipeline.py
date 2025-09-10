import pandas as pd
import numpy as np
from feature_engineering import build_model_input
import config
import os

def load_data(subfolder: str, filename: str, data_dir: str = "../data") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, subfolder, filename))
    df = df.dropna()
    if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
    df = df.loc[(df['cycle number'] != 0)].copy()

    return df

def load_and_prepare_data(
    data_folder=None, 
    cycle_range=None, 
    train_channels=None, 
    test_channels=None, 
    frequency_selection=None
):
    """
    Load and prepare training and testing data with flexible configuration.
    
    Args:
        data_folder (str): Folder containing the CSV files. If None, uses DEFAULT_DATA_FOLDER
        cycle_range (tuple): Optional (start_cycle, end_cycle) to filter data. If None, uses all cycles
        train_channels (list): List of channels to use for training. If None, uses config.TRAIN_CHANNELS
        test_channels (list): List of channels to use for testing. If None, uses config.TEST_CHANNELS
        frequency_selection (str): Optional frequency selection method ('physics', 'correlation', 'combined', or None)
        
    Returns:
        tuple: (X_train, X_test, y_train, y_test)
    """
    # Set defaults
    if data_folder is None: data_folder = config.DEFAULT_DATA_FOLDER
    if train_channels is None: train_channels = config.TRAIN_CHANNELS
    if test_channels is None: test_channels = config.TEST_CHANNELS
    
    # Load data for all required channels
    all_required_channels = list(set(train_channels + test_channels))
    all_channels_data = load_channels_data(data_folder, all_required_channels, cycle_range)
    
    # Split into train/test based on configuration
    train_channels_data = {ch: all_channels_data[ch] for ch in train_channels if ch in all_channels_data}
    test_channels_data = {ch: all_channels_data[ch] for ch in test_channels if ch in all_channels_data}
    
    # Validate we have data
    if not train_channels_data:
        cycle_info = f" in cycle range {cycle_range[0]}-{cycle_range[1]}" if cycle_range else ""
        raise ValueError(f"No training data found for channels {train_channels}{cycle_info}")
    if not test_channels_data:
        cycle_info = f" in cycle range {cycle_range[0]}-{cycle_range[1]}" if cycle_range else ""
        raise ValueError(f"No testing data found for channels {test_channels}{cycle_info}")
    
    # Prepare features and targets
    X_train, X_test, y_train, y_test = prepare_features_and_targets(train_channels_data, test_channels_data, frequency_selection)
    
    # Print info about features
    feature_info = ""
    if frequency_selection:
        feature_info = f" (using {frequency_selection} frequency selection)"
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}{feature_info}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}{feature_info}")
    print(f"Capacity ranges - Train: {y_train.min():.1f}-{y_train.max():.1f}, Test: {y_test.min():.1f}-{y_test.max():.1f}")
    
    return X_train, X_test, y_train, y_test

def load_channels_data(data_folder, channels, cycle_range=None):
    data_by_channel = {}
    for channel in channels:
        df = load_data(data_folder, f"{channel}.csv")
        # Apply cycle range filter if specified
        if cycle_range is not None:
            start_cycle, end_cycle = cycle_range
            df = df[(df['cycle number'] >= start_cycle) & (df['cycle number'] <= end_cycle)].copy()
        
        data_by_channel[channel] = df
    
    return data_by_channel

def prepare_features_and_targets(train_channels_data, test_channels_data, frequency_selection=None):
    """
    Convert channel data to feature matrices and target vectors.
    """
    # Prepare training data
    train_dfs = list(train_channels_data.values())
    test_dfs = list(test_channels_data.values())
    
    # Concatenate DataFrames
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    # Build feature matrices with optional frequency selection
    X_train, y_train = build_model_input(df_train, frequency_selection=frequency_selection)
    X_test, y_test = build_model_input(df_test, frequency_selection=frequency_selection)
    
    return X_train, X_test, y_train, y_test


