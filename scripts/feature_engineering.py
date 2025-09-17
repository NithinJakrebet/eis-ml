import numpy as np
import pandas as pd

def minmax_normalize(arr):
  # Compute the minimum and maximum ignoring NaNs.
  a_min = np.nanmin(arr)
  a_max = np.nanmax(arr)
  # Avoid division by zero: if all values are equal, return zeros.
  if a_max != a_min: return (arr - a_min) / (a_max - a_min)
  else: return np.zeros_like(arr)
  
def build_state_vector(df: pd.DataFrame, selected_frequencies=None):
  """
  For each cycle, extract impedance values at frequencies,
  then min‑max normalize.
  """
  # Keep only valid EIS rows with frequency filtering
  df = (
    df.dropna()
      .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
      .query("Ns in [1,6]")
  )
  
  # Use selected frequencies or all frequencies
  if selected_frequencies is None:
    frequencies_to_use = sorted(df['freq/Hz'].unique())
  else:
    frequencies_to_use = selected_frequencies
  
  valid_cycles = sorted(df['cycle number'].unique())
  state_vectors = []
  
  for cycle in valid_cycles:
    # Extract this cycle's data
    sub = df[df['cycle number'] == cycle].copy()
    # Sort by frequency to ensure consistent ordering
    sub = sub.sort_values('freq/Hz')
    
    # Extract impedance values at specified frequencies in order
    real_z = []
    imag_z = []
    
    for freq in frequencies_to_use:
      freq_data = sub[sub['freq/Hz'] == freq]
      if not freq_data.empty:
        real_z.append(freq_data['Re(Z)/Ohm'].iloc[0])
        imag_z.append(freq_data['Im(Z)/Ohm'].iloc[0])
      else:
        # Fill missing with zeros for selected frequencies
        real_z.append(0.0)
        imag_z.append(0.0)
    
    # Min‑max normalize each
    real_norm = minmax_normalize(np.array(real_z))
    imag_norm = minmax_normalize(np.array(imag_z))
    
    # Concatenate and record
    state_vectors.append(np.concatenate([real_norm, imag_norm]))
  
  return state_vectors, valid_cycles

def build_action_vector(df: pd.DataFrame):
  # Compute the max Q charge for each cycle
  max_charge    = df.groupby('cycle number')['Q charge/mA.h'].max()
  # Compute the max Q discharge for each cycle
  max_discharge = df.groupby('cycle number')['Q discharge/mA.h'].max()
  # Extract the sorted list of cycle IDs
  cycle_numbers    = np.array(sorted(max_charge.index))
  # Re‑index each Series to that sorted order, then grab the numpy values
  charge_values    = max_charge.reindex(cycle_numbers).values
  discharge_values = max_discharge.reindex(cycle_numbers).values
  # Stack into an (n_cycles x 2) matrix: [ [Qcharge₁, Qdischarge₁], [Qcharge₂, Qdischarge₂], … ]
  action_matrix     = np.vstack([charge_values, discharge_values]).T

  return cycle_numbers, action_matrix


def build_model_input(
  df, 
  cycle_range=None, 
  action_vector=True, 
  selected_frequencies=None, 
  frequency_selection=None
):
  # Apply cycle range filter if specified
  if cycle_range:
    min_cycle, max_cycle = cycle_range
    df = df[df['cycle number'].between(min_cycle, max_cycle)]
  
  # Apply frequency selection if requested
  if frequency_selection:
    selected_frequencies = select_frequencies(df, n_frequencies=3)
  
  # Build state vectors (impedance features) with optional frequency selection
  state_vectors, valid_cycles = build_state_vector(df, selected_frequencies)
  
  # Build action vectors if requested
  if action_vector:
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

def select_frequencies(df: pd.DataFrame, n_frequencies=3):
  """Apply physics-informed frequency selection."""
  frequencies = np.array(sorted(df['freq/Hz'].unique()))
  physics_mask = (frequencies >= 1.0) & (frequencies <= 10.0)
  selected_freqs = frequencies[physics_mask]
  
  if len(selected_freqs) < n_frequencies:
    # Add closest frequencies if not enough in range
    remaining = n_frequencies - len(selected_freqs)
    other_freqs = frequencies[~physics_mask]
    distances = np.abs(other_freqs - 5.0)  # Distance from 5 Hz
    closest_indices = np.argsort(distances)[:remaining]
    selected_freqs = np.concatenate([selected_freqs, other_freqs[closest_indices]])
  else:
    # If we have more than needed, prioritize frequencies closest to data-driven optimal: 3.2, 5.6 Hz
    if len(selected_freqs) > n_frequencies:
      optimal_targets = [3.2, 5.6, 10.0, 2.4, 1.8]  # Based on stability analysis
      
      # Calculate distances to optimal frequencies and select closest
      scores = []
      for freq in selected_freqs:
        min_distance = min(abs(freq - target) for target in optimal_targets)
        scores.append(min_distance)
      
      # Select frequencies with smallest distances to optimal targets
      best_indices = np.argsort(scores)[:n_frequencies]
      selected_freqs = selected_freqs[best_indices]
      
  print(f"Selected frequencies: {sorted(selected_freqs)}")
  return sorted(selected_freqs) 