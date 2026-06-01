import numpy as np
from .state_vector import build_state_vector

def build_model_input(df, cycle_range=None, freqs=None, ns_states=None,
                      freq_range=None, capacity_ns=8, return_layout=False):
    """Build (X, y) from an EIS dataframe.

    ns_states / freq_range select which Ns steps and frequency band form the
    EIS state vector; capacity_ns is the rest/measurement step holding capacity.
    All default to the original PEIS-HC-RT behavior (Ns={1,6}, cap at Ns=8).
    Set return_layout=True to also get the feature-layout metadata dict.
    """

    if cycle_range:
        min_cycle, max_cycle = cycle_range
        df = df[df['cycle number'].between(min_cycle, max_cycle)]

    state_vectors, sample_ids, layout = build_state_vector(
        df, frequencies=freqs, ns_states=ns_states, freq_range=freq_range
    )

    X = np.array(state_vectors)
    y = []

    capacity_df = df[df['Ns'] == capacity_ns]
    if capacity_df.empty:
        raise ValueError(
            f"No rows found at capacity_ns={capacity_ns}. Available Ns: "
            f"{sorted(df['Ns'].dropna().unique().tolist())}. "
            f"Pass the correct capacity step for this dataset via capacity_ns."
        )

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
        if cycle_data.empty:
            raise ValueError(
                f"No capacity (Ns={capacity_ns}) row for channel={channel}, "
                f"cycle={cycle}. The capacity step may not align with the EIS "
                f"cycles for this dataset."
            )
        current_capacity = cycle_data['Capacity/mA.h'].iloc[-1]
        initial_capacity = initial_capacities[channel]
        normalized_capacity = current_capacity / initial_capacity  # SOH as fraction
        y.append(normalized_capacity) 
    
    y = np.array(y)

    if return_layout:
        return X, y, layout
    return X, y