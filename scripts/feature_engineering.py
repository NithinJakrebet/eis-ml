import numpy as np
import pandas as pd

def minmax_normalize(arr):
  # Compute the minimum and maximum ignoring NaNs.
  a_min = np.nanmin(arr)
  a_max = np.nanmax(arr)
  
  # Avoid division by zero: if all values are equal, return zeros.
  if a_max != a_min: return (arr - a_min) / (a_max - a_min)
  else: return np.zeros_like(arr)
  
'''
### Building the Input
#### Make state vector FOR EACH CYCLE

- Build EIS-based state vectors from the given dataframe.
- Returns:
```
"state_vector": np.ndarray (the concatenated real+imag EIS data)
```
#### Action vector Data
- Typically, we treat positive I/mA as charge and negative as discharge.
- Returns a dictionary keyed by cycle, each containing a 6-element np.array: 
- [i_charge_avg, total_charge_time, q_charge, i_discharge_avg, total_discharge_time, q_discharge]

### Combine State and Action vectors
Returns:
```
      X (np.ndarray): 2D array of shape (num_valid_cycles, dim_state + dim_action).
      y (np.ndarray): 1D array of discharge capacities for each cycle.
```
'''  


def build_state_vector(df: pd.DataFrame, n_features=100):
  """
  For each cycle, extract the top N real & imaginary impedance values,
  then min‑max normalize. Returns:
    - state_vectors: list of 1D arrays
    - valid_cycles: list of corresponding cycle numbers
  """
  # 1) Keep only valid EIS rows
  df = (
    df.dropna()
      .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
      .query("Ns in [1,6]")
  )
  
  valid_cycles  = sorted(df['cycle number'].unique())
  state_vectors = []
  
  for cycle in valid_cycles:
    # Extract this cycle’s data
    sub = df[df['cycle number'] == cycle]
    
    # Take top N features from real and imaginary parts
    real_z = sub['Re(Z)/Ohm'].nlargest(n_features).values
    imag_z = sub['Im(Z)/Ohm'].nlargest(n_features).values

    #  Min‑max normalize each
    real_norm = minmax_normalize(real_z)
    imag_norm = minmax_normalize(imag_z)
    
    # Concatenate and record
    state_vectors.append(np.concatenate([real_norm, imag_norm]))
  
  return state_vectors, valid_cycles

def build_action_vector(df: pd.DataFrame):
  """
  For each cycle, extract the maximum Q charge and Q discharge values.

  Returns:
    - cycle_numbers: 1D np.array of sorted cycle IDs
    - action_matrix: 2D np.array of shape (n_cycles, 2) containing [max_charge, max_discharge]
  """
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
  
  # Return the cycle IDs and the corresponding 2‑column action matrix
  return cycle_numbers, action_matrix

def build_model_input(df: pd.DataFrame):
    """
    Combine each cycle’s state vector and action vector into one feature matrix.

    Returns:
      - X: NumPy array of shape (n_cycles, state_dimension + action_dimension)
      - valid_cycles: list of cycle numbers corresponding to rows of X
    """
    # 1) Get all state vectors and their cycle IDs
    state_matrix, state_cycles = build_state_vector(df)
    # 2) Get all action vectors and their cycle IDs
    action_cycles, action_matrix = build_action_vector(df)

    # 3) Build a quick lookup from cycle -> action vector
    action_lookup = {cycle: vec for cycle, vec in zip(action_cycles, action_matrix)}

    X = []
    valid_cycles = []

    # 4) For each cycle that has a state vector
    for cycle, state_vec in zip(state_cycles, state_matrix):
        action_vec = action_lookup.get(cycle)
        if action_vec is None:
            # no action data for this cycle → skip
            continue

        # 5) Concatenate state + action
        features = np.concatenate([state_vec, action_vec])

        X.append(features)
        valid_cycles.append(cycle)

    # 6) Stack into a single array and return
    return np.vstack(X), valid_cycles

def drop_last_cycle(df):
    """Drops the last cycle from a dataframe if it exists."""
    if 'cycle number' in df.columns and not df.empty:
        last_cycle = df['cycle number'].max()
        return df[df['cycle number'] < last_cycle]
    return df