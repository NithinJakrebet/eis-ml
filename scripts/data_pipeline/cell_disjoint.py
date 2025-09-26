"""
Cell-Disjoint Splitting Module

Implementation of cell-disjoint train/test splits following the Nature Communications paper approach.
Trains on some cells and tests on completely different cells.
"""

import pandas as pd
import numpy as np


def create_cell_disjoint_split(all_channels_data, train_cells=None, test_cells=None, 
                              analysis_results=None, random_state=42):
    """
    Create cell-disjoint split following Nature Communications paper approach.
    Train on some cells, test on completely different cells.
    
    Args:
        all_channels_data: Dict of {cell_id: DataFrame}
        train_cells: List of cell IDs for training (if None, use recommendations)
        test_cells: List of cell IDs for testing (if None, use recommendations)
        analysis_results: Results from analyze_cell_capacity_ranges()
        random_state: Random seed
    
    Returns:
        train_df, test_df, metadata
    """
    if train_cells is None or test_cells is None:
        if analysis_results is None:
            raise ValueError("Must provide either explicit cell lists or analysis_results for recommendations")
        
        if not analysis_results['recommendations']['cell_disjoint_viable']:
            raise ValueError("Cell-disjoint split not recommended for this data. " + 
                           " ".join(analysis_results['recommendations']['coverage_issues']))
        
        train_cells = analysis_results['recommendations']['recommended_train_cells']
        test_cells = analysis_results['recommendations']['recommended_test_cells']
    
    # Verify cells exist
    available_cells = set(all_channels_data.keys())
    train_cells = [cell for cell in train_cells if cell in available_cells]
    test_cells = [cell for cell in test_cells if cell in available_cells]
    
    if not train_cells or not test_cells:
        raise ValueError(f"Invalid cell selection. Available: {available_cells}")
    
    print(f"=== Cell-Disjoint Split ===")
    print(f"Training cells: {train_cells}")
    print(f"Testing cells: {test_cells}")
    
    # Create train and test dataframes
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
        'split_type': 'cell_disjoint',
        'train_cells': train_cells,
        'test_cells': test_cells,
        'train_capacity_range': (train_capacities.min(), train_capacities.max()),
        'test_capacity_range': (test_capacities.min(), test_capacities.max()),
        'capacity_distribution': _analyze_capacity_distribution(
            train_capacities.values, test_capacities.values
        )
    }
    
    # Check for coverage issues
    train_min, train_max = metadata['train_capacity_range']
    test_min, test_max = metadata['test_capacity_range']
    
    if test_min > train_max:
        print(f"⚠️  Warning: Test range ({test_min:.1f}-{test_max:.1f}) above train range ({train_min:.1f}-{train_max:.1f})")
    elif test_max < train_min:
        print(f"⚠️  Warning: Test range ({test_min:.1f}-{test_max:.1f}) below train range ({train_min:.1f}-{train_max:.1f})")
    
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