import numpy as np
from .state_vector import build_state_vector

def build_model_input(
    df, 
    cycle_range=None, 
    frequencies_to_use=None,
    components_to_use=None,
    use_delta_capacity=False   
): 
    
    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]
    
    state_vectors, sample_ids = build_state_vector(df, frequencies_to_use, components_to_use)
    
    X = np.array(state_vectors)
    y = []
    
    if use_delta_capacity:
        capacity_lookup = df.groupby(['channel', 'cycle number'])['Capacity/mA.h'].last().to_dict()
        
        for channel, cycle in sample_ids:
            current_capacity = capacity_lookup.get((channel, cycle))
            prev_capacity = capacity_lookup.get((channel, cycle - 1))
            
            if current_capacity is not None and prev_capacity is not None:  delta_capacity = current_capacity - prev_capacity
            else:   delta_capacity = 0.0  
            
            y.append(delta_capacity)

        y = np.array(y)
        
    else:
        for channel, cycle in sample_ids:
            cycle_data = df[(df['channel'] == channel) & (df['cycle number'] == cycle)]
            capacity = cycle_data['Capacity/mA.h'].iloc[-1]
            y.append(capacity)
        
        y = np.array(y)
    
    return X, y