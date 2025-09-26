"""
Action Vector Module

Functions for building action vectors from charge/discharge data.
"""

import numpy as np
import pandas as pd

# Build action vector from charge/discharge data for each cycle.
def build_action_vector(df: pd.DataFrame): 
    # Compute the max Q charge for each cycle
    max_charge = df.groupby('cycle number')['Q charge/mA.h'].max()
    # Compute the max Q discharge for each cycle
    max_discharge = df.groupby('cycle number')['Q discharge/mA.h'].max()
    # Extract the sorted list of cycle IDs
    cycle_numbers = np.array(sorted(max_charge.index))
    # Re‑index each Series to that sorted order, then grab the numpy values
    charge_values = max_charge.reindex(cycle_numbers).values
    discharge_values = max_discharge.reindex(cycle_numbers).values
    # Stack into an (n_cycles x 2) matrix: [ [Qcharge₁, Qdischarge₁], [Qcharge₂, Qdischarge₂], … ]
    action_matrix = np.vstack([charge_values, discharge_values]).T

    return cycle_numbers, action_matrix