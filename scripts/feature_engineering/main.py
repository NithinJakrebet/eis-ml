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
    
    # Build state vectors - now returns (channel, cycle) sample IDs
    state_vectors, sample_ids = build_state_vector(df, frequencies_to_use, components_to_use)
    
    # Build target array by matching capacity to each (channel, cycle) pair
    X = np.array(state_vectors)
    y = []
    
    for channel, cycle in sample_ids:
        # Get capacity for this specific (channel, cycle) - no more channel mixing!
        cycle_data = df[(df['channel'] == channel) & (df['cycle number'] == cycle)]
        if not cycle_data.empty:
            capacity = cycle_data['Capacity/mA.h'].iloc[-1]
            y.append(capacity)
        else:
            # This shouldn't happen if state_vector and capacity are aligned
            raise ValueError(f"No capacity data found for channel={channel}, cycle={cycle}")
    
    y = np.array(y)
    
    return X, y