"""
Cell Capacity Analysis Module

Functions for analyzing individual cell capacity ranges, degradation patterns,
and determining suitability for different train/test splitting strategies.
"""

import pandas as pd
import numpy as np
import os
import sys
from pathlib import Path

# Add scripts directory to path for config import
sys.path.append(str(Path(__file__).parent))
import config


def analyze_cell_capacity_ranges(data_folder, channels=None, cycle_range=None, data_dir="../data"):
    """
    Analyze capacity ranges for each cell to determine suitability for different split strategies.
    
    Returns:
        dict: Analysis results including per-cell ranges, degradation coverage, and recommendations
    """
    if channels is None:
        try:
            channels = config.CHANNELS
        except AttributeError:
            channels = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']  # Default fallback
    
    # cycle_range is optional - if None, use all cycles
    if cycle_range is None:
        try:
            cycle_range = config.CYCLE_RANGE
        except AttributeError:
            cycle_range = None  # Use all available cycles
    
    cell_analysis = {}
    all_capacities = []
    
    print("=== Cell Capacity Range Analysis ===")
    print(f"Analyzing {len(channels)} cells: {channels}")
    
    for channel in channels:
        # Load cell data
        df = pd.read_csv(os.path.join(data_dir, data_folder, f"{channel}.csv"))
        df = df.dropna()
        
        # Apply cycle range filter
        if cycle_range:
            df = df[(df['cycle number'] >= cycle_range[0]) & 
                   (df['cycle number'] <= cycle_range[1])]
        
        # Get capacity per cycle (last measurement in each cycle)
        capacity_per_cycle = df.groupby('cycle number')['Capacity/mA.h'].last()
        
        # Calculate statistics
        min_cap = capacity_per_cycle.min()
        max_cap = capacity_per_cycle.max()
        range_cap = max_cap - min_cap
        mean_cap = capacity_per_cycle.mean()
        std_cap = capacity_per_cycle.std()
        degradation_rate = (max_cap - min_cap) / len(capacity_per_cycle)
        
        cell_analysis[channel] = {
            'min_capacity': min_cap,
            'max_capacity': max_cap,
            'capacity_range': range_cap,
            'mean_capacity': mean_cap,
            'std_capacity': std_cap,
            'degradation_rate': degradation_rate,
            'num_cycles': len(capacity_per_cycle),
            'final_capacity': capacity_per_cycle.iloc[-1],
            'initial_capacity': capacity_per_cycle.iloc[0]
        }
        
        all_capacities.extend(capacity_per_cycle.values)
        
        print(f"{channel}: {min_cap:.1f}-{max_cap:.1f} mAh "
              f"(range: {range_cap:.1f}, degradation: {degradation_rate:.2f} mAh/cycle)")
    
    # Calculate global statistics
    global_min = min(all_capacities)
    global_max = max(all_capacities)
    global_range = global_max - global_min
    
    print(f"\nGlobal capacity range: {global_min:.1f}-{global_max:.1f} mAh")
    
    # Identify overlap regions and recommendations
    recommendations = _generate_split_recommendations(cell_analysis, global_min, global_max)
    
    return {
        'cell_analysis': cell_analysis,
        'global_min': global_min,
        'global_max': global_max,
        'global_range': global_range,
        'recommendations': recommendations
    }


def _generate_split_recommendations(cell_analysis, global_min, global_max):
    """Generate recommendations for train/test splits based on capacity analysis."""
    recommendations = {
        'cell_disjoint_viable': True,
        'recommended_train_cells': [],
        'recommended_test_cells': [],
        'coverage_issues': [],
        'stratified_recommendations': {}
    }
    
    # Define capacity regions
    low_threshold = global_min + (global_max - global_min) * 0.25
    high_threshold = global_min + (global_max - global_min) * 0.75
    
    # Categorize cells by their coverage
    low_capacity_cells = []
    mid_capacity_cells = []
    high_capacity_cells = []
    
    for cell, stats in cell_analysis.items():
        min_cap = stats['min_capacity']
        max_cap = stats['max_capacity']
        
        if min_cap <= low_threshold:
            low_capacity_cells.append(cell)
        if min_cap <= high_threshold and max_cap >= low_threshold:
            mid_capacity_cells.append(cell)
        if max_cap >= high_threshold:
            high_capacity_cells.append(cell)
    
    # Check for sufficient coverage overlap
    if not low_capacity_cells or not high_capacity_cells:
        recommendations['cell_disjoint_viable'] = False
        recommendations['coverage_issues'].append("Insufficient coverage range")
    
    # Check if we can create meaningful train/test splits
    if len(low_capacity_cells) < 2 or len(high_capacity_cells) < 2:
        recommendations['cell_disjoint_viable'] = False
        recommendations['coverage_issues'].append("Insufficient cells for proper split")
    
    # Generate cell assignments if viable
    if recommendations['cell_disjoint_viable']:
        # Put degraded cells in training (to learn degradation patterns)
        recommendations['recommended_train_cells'] = (low_capacity_cells + 
                                                    mid_capacity_cells[:len(mid_capacity_cells)//2])
        recommendations['recommended_test_cells'] = (high_capacity_cells + 
                                                   mid_capacity_cells[len(mid_capacity_cells)//2:])
    
    # Stratified recommendations
    recommendations['stratified_recommendations'] = {
        'suggested_bins': min(10, len(cell_analysis)),
        'balance_degradation': True,
        'include_all_cells': True
    }
    
    return recommendations