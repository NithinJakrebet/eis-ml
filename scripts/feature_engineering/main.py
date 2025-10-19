import numpy as np
from .state_vector import build_state_vector

def build_model_input(
    df, 
    cycle_range=None, 
    frequencies_to_use=None,
    components_to_use=None
): 
    
    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]
    
    state_vectors, sample_ids = build_state_vector(df, frequencies_to_use, components_to_use)
    
    X = np.array(state_vectors)
    y = []
    
    for channel, cycle in sample_ids:
        cycle_data = df[(df['channel'] == channel) & (df['cycle number'] == cycle)]
        capacity = cycle_data['Capacity/mA.h'].iloc[-1]
        y.append(capacity)

    
    y = np.array(y)
    
    return X, y