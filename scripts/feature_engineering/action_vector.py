import numpy as np
import pandas as pd

# Build action vector from charge/discharge data for each cycle.
def build_action_vector(df: pd.DataFrame): 
    max_charge = df.groupby('cycle number')['Q charge/mA.h'].max()
    max_discharge = df.groupby('cycle number')['Q discharge/mA.h'].max()
    cycle_numbers = np.array(sorted(max_charge.index))
    charge_values = max_charge.reindex(cycle_numbers).values
    discharge_values = max_discharge.reindex(cycle_numbers).values
    action_matrix = np.vstack([charge_values, discharge_values]).T

    return cycle_numbers, action_matrix