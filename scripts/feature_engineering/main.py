import numpy as np
from .state_vector import build_state_vector

def build_model_input(df, cycle_range=None, freqs=None): 
    
    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]
    
    state_vectors, sample_ids = build_state_vector(df,frequencies=freqs)
    
    X = np.array(state_vectors)
    y = []
        
    capacity_df = df[df['Ns'] == 8]
    
    # Get initial capacity for each channel (first cycle)
    initial_capacities = {}
    for channel in capacity_df['channel'].unique():
        channel_data = capacity_df[capacity_df['channel'] == channel]
        min_cycle = channel_data['cycle number'].min()
        initial_cycle_data = channel_data[channel_data['cycle number'] == min_cycle]
        if not initial_cycle_data.empty:
            initial_capacities[channel] = initial_cycle_data['Capacity/mA.h'].iloc[-1]
    
    # Calculate normalized capacity for each sample
    for channel, cycle in sample_ids:
        cycle_data = capacity_df[(capacity_df['channel'] == channel) & (capacity_df['cycle number'] == cycle)]
        current_capacity = cycle_data['Capacity/mA.h'].iloc[-1]
        initial_capacity = initial_capacities[channel]
        normalized_capacity = current_capacity / initial_capacity  # SOH as fraction
        y.append(normalized_capacity) 
    
    y = np.array(y)
    
    return X, y