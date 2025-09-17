import pandas as pd
import numpy as np
from feature_engineering import build_model_input
import config
import os

def load_and_prepare_data(
    data_folder=None, 
    cycle_range=None,
    frequency_selection=None,
    include_action_vector=True
):
    if data_folder is None: data_folder = config.DEFAULT_DATA_FOLDER
    train_channels, test_channels = config.TRAIN_CHANNELS, config.TEST_CHANNELS
    
    # Load data for all required channels
    all_required_channels = list(set(train_channels + test_channels))
    all_channels_data = load_channels_data(data_folder, all_required_channels, cycle_range)
    
    # Split into train/test based on configuration
    train_channels_data = {ch: all_channels_data[ch] for ch in train_channels if ch in all_channels_data}
    test_channels_data = {ch: all_channels_data[ch] for ch in test_channels if ch in all_channels_data}
    
    # Prepare features and targets
    X_train, X_test, y_train, y_test = prepare_features_and_targets(
        train_channels_data, 
        test_channels_data, 
        frequency_selection,
        include_action_vector
    )
    
    # Print info about features
    feature_info = ""
    if frequency_selection:
        feature_info = " with frequency selection"
    if not include_action_vector:
        feature_info += " (state vector only)"
    
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}{feature_info}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}{feature_info}")
    print(f"Capacity ranges - Train: {y_train.min():.1f}-{y_train.max():.1f}, Test: {y_test.min():.1f}-{y_test.max():.1f}")
    
    return X_train, X_test, y_train, y_test

def load_channels_data(data_folder, channels, cycle_range=None, data_dir="../data"):
    data_by_channel = {}
    for channel in channels:
        # Load CSV file
        df = pd.read_csv(os.path.join(data_dir, data_folder, f"{channel}.csv"))
        df = df.dropna()
        if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
        
        # Apply cycle range filter if specified
        if cycle_range is not None:
            start_cycle, end_cycle = cycle_range
            df = df[(df['cycle number'] >= start_cycle) & (df['cycle number'] <= end_cycle)].copy()
        
        data_by_channel[channel] = df
    
    return data_by_channel

def prepare_features_and_targets(train_channels_data, test_channels_data, frequency_selection=None, include_action_vector=True):
    # Prepare training data
    train_dfs = list(train_channels_data.values())
    test_dfs = list(test_channels_data.values())
    
    # Concatenate DataFrames
    df_train = pd.concat(train_dfs, ignore_index=True)
    df_test = pd.concat(test_dfs, ignore_index=True)
    
    # Build feature matrices with optional frequency selection and action vector
    X_train, y_train = build_model_input(df_train, frequency_selection=frequency_selection, include_action_vector=include_action_vector)
    X_test, y_test = build_model_input(df_test, frequency_selection=frequency_selection, include_action_vector=include_action_vector)
    
    return X_train, X_test, y_train, y_test


