"""
Main Feature Engineering Module

Primary feature building and preprocessing functions that orchestrate 
state vectors, action vectors, and other feature engineering components.
"""

import numpy as np
import pandas as pd
from .state_vector import build_state_vector
from .action_vector import build_action_vector
from .frequency_selection import select_frequencies


def build_model_input(
    df, 
    cycle_range=None, 
    include_action_vector=True, 
    selected_frequencies=None, 
    frequency_selection=None
): #  Build complete feature matrix and target vector from EIS data.
    
    # Apply cycle range filter if specified
    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]
    
    # Apply frequency selection if requested
    if frequency_selection: selected_frequencies = select_frequencies(df, n_frequencies=3)

    # Build state vectors (impedance features) with optional frequency selection
    state_vectors, valid_cycles = build_state_vector(df, selected_frequencies)
    
    # Build action vectors if requested
    if include_action_vector:
        action_cycles, action_matrix = build_action_vector(df)
        
        # Create lookup for action vectors by cycle
        action_lookup = {cycle: action_matrix[i] for i, cycle in enumerate(action_cycles)}
        
        # Combine state and action vectors for matching cycles
        X_list = []
        y_list = []
        
        for i, cycle in enumerate(valid_cycles):
            if cycle in action_lookup:
                state_vec = state_vectors[i]
                action_vec = action_lookup[cycle]
                X_list.append(np.concatenate([state_vec, action_vec]))
                
                # Extract capacity target for this cycle
                cycle_data = df[df['cycle number'] == cycle]
                if not cycle_data.empty:
                    capacity = cycle_data['Capacity/mA.h'].iloc[-1]  # Take last value for cycle
                    y_list.append(capacity)
        
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        # Use only state vectors 
        X = np.array(state_vectors)
        # Extract capacity targets for all valid cycles
        y = []
        for cycle in valid_cycles:
            cycle_data = df[df['cycle number'] == cycle]
            if not cycle_data.empty:
                capacity = cycle_data['Capacity/mA.h'].iloc[-1]
                y.append(capacity)
        y = np.array(y)
    
    return X, y