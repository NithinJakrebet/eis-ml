import numpy as np
import pandas as pd

def build_action_vector(df: pd.DataFrame):
    cycles = np.sort(df['cycle number'].unique())
    rows = []

    for c in cycles:
        d = df[df['cycle number'] == c]

        charge_rows = d[d['Ns'].isin([3])]
        discharge_rows = d[d['Ns'].isin([8])]

        I_charge = (np.median(np.abs(charge_rows['I/mA'].values))
                    if not charge_rows.empty else np.nan)
        I_discharge = (np.median(np.abs(discharge_rows['I/mA'].values))
                       if not discharge_rows.empty else np.nan)

        rows.append([float(I_charge), float(I_discharge)])

    return cycles, np.asarray(rows, dtype=float)
