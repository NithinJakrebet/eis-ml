"""
Stratified Fold Module

Enhanced stratified splitting strategies with automatic bin selection and group-aware folding
for EIS battery degradation data.
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedGroupKFold


def create_improved_stratified_split(all_channels_data, test_size=0.2, min_samples_per_bin=20, 
                                   random_state=42):
    """
    Enhanced stratified group split that automatically determines optimal bin count
    and ensures each capacity class appears in multiple cells.
    
    Args:
        all_channels_data: Dict of {cell_id: DataFrame}
        test_size: Fraction for test set
        min_samples_per_bin: Minimum samples required per capacity bin
        random_state: Random seed
    
    Returns:
        train_df, test_df, metadata
    """
    # Combine all data with cell metadata
    combined_df = pd.concat([
        df.assign(cell_id=cell_id) 
        for cell_id, df in all_channels_data.items()
    ], ignore_index=True)
    
    # Extract capacity for stratification
    capacity_per_cycle = combined_df.groupby(['cell_id', 'cycle number'])['Capacity/mA.h'].last().reset_index()
    
    # Automatically determine optimal number of bins
    total_samples = len(capacity_per_cycle)
    n_cells = len(all_channels_data)
    n_splits = int(1 / test_size)
    
    # Start with many bins and reduce until we have stable stratification
    optimal_bins = None
    for n_bins in range(min(20, total_samples // min_samples_per_bin), 2, -1):
        try:
            # Create capacity bins
            temp_df = capacity_per_cycle.copy()
            temp_df['capacity_bin'] = pd.qcut(
                temp_df['Capacity/mA.h'], 
                q=n_bins, 
                duplicates='drop'
            )
            
            # Check if each bin appears in multiple cells
            bin_cell_counts = temp_df.groupby('capacity_bin')['cell_id'].nunique()
            min_cells_per_bin = bin_cell_counts.min()
            
            # Check if we can do proper stratification
            if min_cells_per_bin >= n_splits:
                optimal_bins = n_bins
                break
                
        except Exception:
            continue
    
    if optimal_bins is None:
        print("Warning: Could not find optimal stratified split. Falling back to simple group split.")
        return _fallback_to_group_split(all_channels_data, test_size, random_state)
    
    print(f"Using {optimal_bins} capacity bins for stratified split")
    
    # Create the final stratified split
    capacity_per_cycle = capacity_per_cycle.copy()
    capacity_per_cycle['capacity_bin'] = pd.qcut(
        capacity_per_cycle['Capacity/mA.h'], 
        q=optimal_bins, 
        duplicates='drop'
    )
    
    # Use StratifiedGroupKFold
    sgkf = StratifiedGroupKFold(
        n_splits=n_splits, 
        shuffle=True, 
        random_state=random_state
    )
    
    # Get stratification targets and groups
    y_strat = capacity_per_cycle['capacity_bin'].cat.codes.values
    groups = capacity_per_cycle['cell_id'].values
    
    # Get first split (train/test)
    train_idx, test_idx = next(sgkf.split(capacity_per_cycle, y_strat, groups))
    
    train_meta = capacity_per_cycle.iloc[train_idx]
    test_meta = capacity_per_cycle.iloc[test_idx]
    
    # Create train/test cell cycle pairs
    train_pairs = set(zip(train_meta['cell_id'], train_meta['cycle number']))
    test_pairs = set(zip(test_meta['cell_id'], test_meta['cycle number']))
    
    # Filter data based on pairs
    combined_df['is_train'] = combined_df.apply(
        lambda row: (row['cell_id'], row['cycle number']) in train_pairs, axis=1
    )
    
    train_df = combined_df[combined_df['is_train']].drop('is_train', axis=1)
    test_df = combined_df[~combined_df['is_train']].drop('is_train', axis=1)
    
    # Calculate metadata
    train_capacities = train_meta['Capacity/mA.h'].values
    test_capacities = test_meta['Capacity/mA.h'].values
    
    metadata = {
        'split_type': 'improved_stratified',
        'n_capacity_bins': optimal_bins,
        'train_cells': sorted(train_meta['cell_id'].unique()),
        'test_cells': sorted(test_meta['cell_id'].unique()),
        'train_capacity_range': (train_capacities.min(), train_capacities.max()),
        'test_capacity_range': (test_capacities.min(), test_capacities.max()),
        'capacity_distribution': _analyze_capacity_distribution(train_capacities, test_capacities)
    }
    
    return train_df, test_df, metadata


def create_stratified_split(capacity_df, test_size=0.2, n_bins=10, random_state=42):
    """
    Create stratified group split ensuring balanced capacity distribution.
    """
    # Create capacity bins for stratification
    capacity_df = capacity_df.copy()
    capacity_df['capacity_bin'] = pd.qcut(
        capacity_df['Capacity/mA.h'], 
        q=n_bins, 
        duplicates='drop'
    )
    
    # Check if we have enough groups for stratification
    group_counts = capacity_df.groupby(['cell_id', 'capacity_bin']).size().reset_index(name='count')
    cells_per_bin = group_counts.groupby('capacity_bin')['cell_id'].nunique()
    
    min_cells_per_bin = cells_per_bin.min()
    n_splits = int(1 / test_size)
    
    if min_cells_per_bin >= n_splits:
        # Use StratifiedGroupKFold for proper stratification
        sgkf = StratifiedGroupKFold(
            n_splits=n_splits, 
            shuffle=True, 
            random_state=random_state
        )
        
        # Get stratification targets and groups
        y_strat = capacity_df['capacity_bin'].cat.codes.values
        groups = capacity_df['cell_id'].values
        
        # Get first split (train/test)
        train_idx, test_idx = next(sgkf.split(capacity_df, y_strat, groups))
        
        train_meta = capacity_df.iloc[train_idx]
        test_meta = capacity_df.iloc[test_idx]
        
        return train_meta, test_meta
    else:
        # Fallback to grouped split if stratification isn't possible
        print(f"Warning: Not enough cells per capacity bin for stratification. Using grouped split.")
        cells = capacity_df['cell_id'].unique()
        np.random.seed(random_state)
        np.random.shuffle(cells)
        
        n_test_cells = max(1, int(len(cells) * test_size))
        test_cells = cells[:n_test_cells]
        train_cells = cells[n_test_cells:]
        
        train_meta = capacity_df[capacity_df['cell_id'].isin(train_cells)]
        test_meta = capacity_df[capacity_df['cell_id'].isin(test_cells)]
        
        return train_meta, test_meta


def _fallback_to_group_split(all_channels_data, test_size=0.2, random_state=42):
    """Fallback to simple group-based split when stratification fails."""
    cells = list(all_channels_data.keys())
    np.random.seed(random_state)
    np.random.shuffle(cells)
    
    n_test_cells = max(1, int(len(cells) * test_size))
    test_cells = cells[:n_test_cells]
    train_cells = cells[n_test_cells:]
    
    train_dfs = []
    test_dfs = []
    
    for cell_id in train_cells:
        df = all_channels_data[cell_id].copy()
        df['cell_id'] = cell_id
        train_dfs.append(df)
    
    for cell_id in test_cells:
        df = all_channels_data[cell_id].copy()
        df['cell_id'] = cell_id
        test_dfs.append(df)
    
    train_df = pd.concat(train_dfs, ignore_index=True)
    test_df = pd.concat(test_dfs, ignore_index=True)
    
    # Calculate metadata
    train_capacities = train_df.groupby(['cell_id', 'cycle number'])['Capacity/mA.h'].last()
    test_capacities = test_df.groupby(['cell_id', 'cycle number'])['Capacity/mA.h'].last()
    
    metadata = {
        'split_type': 'fallback_group',
        'train_cells': train_cells,
        'test_cells': test_cells,
        'train_capacity_range': (train_capacities.min(), train_capacities.max()),
        'test_capacity_range': (test_capacities.min(), test_capacities.max()),
        'capacity_distribution': _analyze_capacity_distribution(
            train_capacities.values, test_capacities.values
        )
    }
    
    return train_df, test_df, metadata


def _analyze_capacity_distribution(train_capacities, test_capacities):
    """Analyze the distribution of capacities between train and test sets."""
    return {
        'train_mean': np.mean(train_capacities),
        'train_std': np.std(train_capacities),
        'test_mean': np.mean(test_capacities),
        'test_std': np.std(test_capacities),
        'overlap_min': max(np.min(train_capacities), np.min(test_capacities)),
        'overlap_max': min(np.max(train_capacities), np.max(test_capacities))
    }