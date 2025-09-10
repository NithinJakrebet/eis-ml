import numpy as np
import pandas as pd

def minmax_normalize(arr):
  # Compute the minimum and maximum ignoring NaNs.
  a_min = np.nanmin(arr)
  a_max = np.nanmax(arr)
  # Avoid division by zero: if all values are equal, return zeros.
  if a_max != a_min: return (arr - a_min) / (a_max - a_min)
  else: return np.zeros_like(arr)
  
def build_state_vector(df: pd.DataFrame):
  """
  For each cycle, extract impedance values at the 69 standard measured frequencies,
  then min‑max normalize.
  """
    # Keep only valid EIS rows with frequency filtering
  df = (
    df.dropna()
      .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
      .query("Ns in [1,6]")
  )
  
  # Get the standard frequency grid (69 frequencies)
  valid_cycles = sorted(df['cycle number'].unique())
  state_vectors = []
  
  for cycle in valid_cycles:
    # Extract this cycle's data
    sub = df[df['cycle number'] == cycle].copy()
    # Sort by frequency to ensure consistent ordering
    sub = sub.sort_values('freq/Hz')
    
    # Extract impedance values at standard frequencies in order
    real_z = []
    imag_z = []
    
    for freq in sorted(df['freq/Hz'].unique()):
      freq_data = sub[sub['freq/Hz'] == freq]
      if not freq_data.empty:
        real_z.append(freq_data['Re(Z)/Ohm'].iloc[0])
        imag_z.append(freq_data['Im(Z)/Ohm'].iloc[0])
    
    # Min‑max normalize each
    real_norm = minmax_normalize(np.array(real_z))
    imag_norm = minmax_normalize(np.array(imag_z))
    
    # Concatenate and record (69 real + 69 imaginary = 138 features)
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


def build_model_input(df: pd.DataFrame):
    """
    Combine each cycle’s state vector and action vector into one feature matrix.
    """
    # Get all state vectors and their cycle IDs
    state_matrix, state_cycles = build_state_vector(df)
    # Get all action vectors and their cycle IDs
    action_cycles, action_matrix = build_action_vector(df)
    # Build a quick lookup from cycle -> action vector
    action_lookup = {cycle: vec for cycle, vec in zip(action_cycles, action_matrix)}

    X = []
    valid_cycles = []

    # For each cycle that has a state vector
    for cycle, state_vec in zip(state_cycles, state_matrix):
      action_vec = action_lookup.get(cycle)
      features = np.concatenate([state_vec, action_vec])
      X.append(features)
      valid_cycles.append(cycle)

    # Stack into a single array and return
    return np.vstack(X), valid_cycles

# def drop_last_cycle(df):
#     """Drops the last cycle from a dataframe if it exists."""
#     if 'cycle number' in df.columns and not df.empty:
#         last_cycle = df['cycle number'].max()
#         return df[df['cycle number'] < last_cycle]
#     return df