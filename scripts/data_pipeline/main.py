"""
Main Data Pipeline Module

Primary data loading, preprocessing, and preparation functions for EIS battery analysis.
This module orchestrates the various splitting strategies and provides the main interface.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Import config from parent scripts directory
sys.path.append(str(Path(__file__).parent))

import config
from feature_engineering.main import build_model_input

from .cell_analysis import analyze_cell_capacity_ranges
from .cell_disjoint import create_cell_disjoint_split
from .stratified_fold import create_improved_stratified_split, create_stratified_split


def load_and_prepare_data(data_folder=None, frequency_selection=None, include_action_vector=True, 
                         split_method='original', test_size=0.2, n_capacity_bins=10, 
                         cycle_range=None, train_cells=None, test_cells=None, random_state=42):
    """
    Load EIS data and apply the specified train/test splitting strategy.
    
    Args:
        data_folder: Name of data folder (default from config)
        frequency_selection: Array of selected frequencies or None for physics selection
        include_action_vector: Whether to include action features
        split_method: 'original', 'stratified', 'improved_stratified', 'cell_disjoint', 'analyze_cells'
        test_size: Fraction for test set (ignored for 'original' and 'cell_disjoint')
        n_capacity_bins: Number of capacity bins for stratification
        cycle_range: Optional tuple (start_cycle, end_cycle) to filter cycles
        train_cells: Explicit list of cells for training (cell_disjoint only)
        test_cells: Explicit list of cells for testing (cell_disjoint only)
        random_state: Random seed
    
    Returns:
        If split_method='analyze_cells': analysis_results dict
        Otherwise: X_train, X_test, y_train, y_test, metadata (if using new methods)
                  or X_train, X_test, y_train, y_test (if using original method)
    """
    if data_folder is None: data_folder = config.DEFAULT_DATA_FOLDER

    
    # Handle cell analysis only
    if split_method == 'analyze_cells': return analyze_cell_capacity_ranges(data_folder, cycle_range=cycle_range)
    
    # Get all available channels
    if split_method == 'original': all_channels = list(set(config.TRAIN_CHANNELS + config.TEST_CHANNELS))
    else: all_channels = config.CHANNELS  # Use all available channels

    # Load all data
    all_channels_data = load_channels_data(data_folder, all_channels, cycle_range)
    
    # Apply the chosen split method using match-case (Python 3.10+)
    match split_method:
        case 'original':
            return _apply_original_split(all_channels_data, frequency_selection, include_action_vector)
        case 'stratified':
            train_df, test_df, train_meta, test_meta = _create_stratified_group_split(
                all_channels_data, test_size, n_capacity_bins, random_state
            )
            metadata = _create_metadata_from_legacy_split(train_meta, test_meta, 'stratified')
        case 'improved_stratified':
            train_df, test_df, metadata = create_improved_stratified_split(
                all_channels_data, test_size, random_state=random_state
            )
        case 'cell_disjoint':
            # Get analysis results if cells not specified
            analysis_results = None
            if train_cells is None or test_cells is None:
                analysis_results = analyze_cell_capacity_ranges(
                    data_folder, all_channels, cycle_range
                )
            train_df, test_df, metadata = create_cell_disjoint_split(
                all_channels_data, train_cells, test_cells, analysis_results, random_state
            )
        case _:
            raise ValueError(f"Unknown split_method: {split_method}")
    
    # Prepare features and targets for new methods
    X_train, y_train = build_model_input(
        train_df, 
        frequency_selection=frequency_selection, 
        include_action_vector=include_action_vector
    )
    X_test, y_test = build_model_input(
        test_df, 
        frequency_selection=frequency_selection, 
        include_action_vector=include_action_vector
    )
    
    # Update metadata with final ranges
    metadata['train_capacity_range'] = (y_train.min(), y_train.max())
    metadata['test_capacity_range'] = (y_test.min(), y_test.max())
    
    # Print distribution analysis
    print_distribution_analysis(X_train, X_test, y_train, y_test, metadata, frequency_selection)
    
    return X_train, X_test, y_train, y_test, metadata


def load_channels_data(data_folder, channels, cycle_range=None, data_dir="../../data"):
    """Load data for specified channels with optional cycle filtering."""
    data_by_channel = {}
    for channel in channels:
        # Load CSV file
        df = pd.read_csv(os.path.join(data_dir, data_folder, f"{channel}.csv"))
        df = df.dropna()
        if "#NAME?" in df.columns: 
            df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
        
        # Apply cycle range filter if specified
        if cycle_range is not None:
            start_cycle, end_cycle = cycle_range
            df = df[(df['cycle number'] >= start_cycle) & (df['cycle number'] <= end_cycle)].copy()
        
        data_by_channel[channel] = df
    
    return data_by_channel


def prepare_features_and_targets(train_channels_data, test_channels_data, frequency_selection=None, include_action_vector=True):
    """Prepare feature matrices from channel data dictionaries."""
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


def print_distribution_analysis(X_train, X_test, y_train, y_test, metadata, frequency_selection):
    """Print comprehensive distribution analysis."""
    feature_info = ""
    if frequency_selection:
        feature_info = " with frequency selection"
    
    print(f"=== {metadata['split_type'].title()} Split Results ===")
    print(f"X_train: {X_train.shape}, y_train: {y_train.shape}{feature_info}")
    print(f"X_test: {X_test.shape}, y_test: {y_test.shape}{feature_info}")
    print(f"Train capacity range: {metadata['train_capacity_range'][0]:.1f}-{metadata['train_capacity_range'][1]:.1f}")
    print(f"Test capacity range: {metadata['test_capacity_range'][0]:.1f}-{metadata['test_capacity_range'][1]:.1f}")
    print(f"Train cells: {metadata['train_cells']}")
    print(f"Test cells: {metadata['test_cells']}")
    
    # Check for coverage gaps if distribution is available
    if 'capacity_distribution' in metadata:
        distribution = metadata['capacity_distribution']
        if isinstance(distribution, pd.DataFrame):
            zero_train_bins = distribution[distribution['train_count'] == 0]
            zero_test_bins = distribution[distribution['test_count'] == 0]
            
            if len(zero_train_bins) > 0:
                print(f"⚠️  Warning: {len(zero_train_bins)} bins have no training data")
            if len(zero_test_bins) > 0:
                print(f"⚠️  Warning: {len(zero_test_bins)} bins have no test data")
            
            print("\n=== Capacity Distribution ===")
            print(distribution.round(3))


# Legacy functions for backwards compatibility
def _apply_original_split(all_channels_data, frequency_selection, include_action_vector):
    """Apply the original predefined train/test split."""
    try:
        train_channels, test_channels = config.TRAIN_CHANNELS, config.TEST_CHANNELS
    except AttributeError:
        # Default fallback split
        train_channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
        test_channels = ['A7', 'A8']
    
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


def _create_stratified_group_split(all_channels_data, test_size=0.2, n_bins=10, random_state=42):
    """Legacy stratified split function."""
    # Combine all data with cell metadata
    combined_df = pd.concat([
        df.assign(cell_id=cell_id) 
        for cell_id, df in all_channels_data.items()
    ], ignore_index=True)
    
    # Extract capacity for stratification
    capacity_per_cycle = combined_df.groupby(['cell_id', 'cycle number'])['Capacity/mA.h'].last().reset_index()
    
    # Use the stratified fold function
    train_meta, test_meta = create_stratified_split(capacity_per_cycle, test_size, n_bins, random_state)
    
    # Create train/test cell cycle pairs
    train_pairs = set(zip(train_meta['cell_id'], train_meta['cycle number']))
    test_pairs = set(zip(test_meta['cell_id'], test_meta['cycle number']))
    
    # Filter data based on pairs
    combined_df['is_train'] = combined_df.apply(
        lambda row: (row['cell_id'], row['cycle number']) in train_pairs, axis=1
    )
    
    train_df = combined_df[combined_df['is_train']].drop('is_train', axis=1)
    test_df = combined_df[~combined_df['is_train']].drop('is_train', axis=1)
    
    return train_df, test_df, train_meta, test_meta


def _create_metadata_from_legacy_split(train_meta, test_meta, split_type):
    """Create metadata dict from legacy stratified split results."""
    train_capacities = train_meta['Capacity/mA.h'].values
    test_capacities = test_meta['Capacity/mA.h'].values
    
    return {
        'split_type': split_type,
        'train_cells': sorted(train_meta['cell_id'].unique()),
        'test_cells': sorted(test_meta['cell_id'].unique()),
        'train_capacity_range': (train_capacities.min(), train_capacities.max()),
        'test_capacity_range': (test_capacities.min(), test_capacities.max()),
        'capacity_distribution': _analyze_capacity_distribution(train_capacities, test_capacities)
    }


def _analyze_capacity_distribution(train_capacities, test_capacities, n_bins=10):
    """Analyze capacity distribution between train and test sets."""
    overall_min = min(train_capacities.min(), test_capacities.min())
    overall_max = max(train_capacities.max(), test_capacities.max())
    
    bins = np.linspace(overall_min, overall_max, n_bins + 1)
    
    train_hist, _ = np.histogram(train_capacities, bins=bins)
    test_hist, _ = np.histogram(test_capacities, bins=bins)
    
    distribution_df = pd.DataFrame({
        'bin_left': bins[:-1],
        'bin_right': bins[1:],
        'train_count': train_hist,
        'test_count': test_hist,
        'train_ratio': train_hist / len(train_capacities),
        'test_ratio': test_hist / len(test_capacities)
    })
    
    return distribution_df