import numpy as np
import pandas as pd
from .normalization import minmax_normalize

def build_state_vector(df: pd.DataFrame, selected_frequencies=None): # For each cycle, extract impedance values at frequencies, then min‑max normalize.
    df = (
        df.dropna()
          .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
          .query("Ns in [1,6]")
    )
    
    frequencies_to_use = selected_frequencies if selected_frequencies is not None else sorted(df['freq/Hz'].unique())
    
    valid_cycles = sorted(df['cycle number'].unique())
    state_vectors = []
    
    for cycle in valid_cycles:
        sub = df[df['cycle number'] == cycle].copy()
        sub = sub.sort_values('freq/Hz')
        
        real_z = []
        imag_z = []
        
        for freq in frequencies_to_use:
            freq_data = sub[sub['freq/Hz'] == freq]
            if not freq_data.empty:
                real_z.append(freq_data['Re(Z)/Ohm'].iloc[0])
                imag_z.append(freq_data['Im(Z)/Ohm'].iloc[0])
            else:
                real_z.append(0.0)
                imag_z.append(0.0)
        
        real_norm = minmax_normalize(np.array(real_z))
        imag_norm = minmax_normalize(np.array(imag_z))
        
        state_vectors.append(np.concatenate([real_norm, imag_norm]))
    
    return state_vectors, valid_cycles