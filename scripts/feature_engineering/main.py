import numpy as np
from .state_vector import build_state_vector
from .action_vector import build_action_vector
from .frequency_selection import get_physics_frequencies

def build_model_input(
    df, 
    cycle_range=None, 
    include_action_vector=True, 
    selected_frequencies=None, 
    frequency_selection=None
): 
    
    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]
    
    if frequency_selection: selected_frequencies = get_physics_frequencies()

    state_vectors, valid_cycles = build_state_vector(df, selected_frequencies)
    
    if include_action_vector:
        action_cycles, action_matrix = build_action_vector(df)
        action_lookup = {cycle: action_matrix[i] for i, cycle in enumerate(action_cycles)}
        
        X_list = []
        y_list = []
        
        for i, cycle in enumerate(valid_cycles):
            if cycle in action_lookup:
                state_vec = state_vectors[i]
                action_vec = action_lookup[cycle]
                X_list.append(np.concatenate([state_vec, action_vec]))
                
                cycle_data = df[df['cycle number'] == cycle]
                if not cycle_data.empty:
                    capacity = cycle_data['Capacity/mA.h'].iloc[-1]  # Take last value for cycle
                    y_list.append(capacity)
        
        X = np.array(X_list)
        y = np.array(y_list)
    else:
        X = np.array(state_vectors)
        y = []
        for cycle in valid_cycles:
            cycle_data = df[df['cycle number'] == cycle]
            if not cycle_data.empty:
                capacity = cycle_data['Capacity/mA.h'].iloc[-1]
                y.append(capacity)
        y = np.array(y)
    
    return X, y