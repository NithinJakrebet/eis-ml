import numpy as np
import pandas as pd

def build_state_vector(
    df: pd.DataFrame,
    frequencies: list = None, 
    ):
    df = (
        df.dropna()
          .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
          .query("Ns in [6]")
    )
    
    if 'channel' not in df.columns: raise ValueError("DataFrame must have a 'channel' column. Use load_single_channel to add it.")
    
    freqs = sorted(df['freq/Hz'].unique()) if frequencies is None else frequencies
        
    channel_cycle_pairs = df.groupby(['channel', 'cycle number'], as_index=False).first()[['channel', 'cycle number']]
    
    state_vectors = []
    sample_ids = []    
    for _, row in channel_cycle_pairs.iterrows():
        channel = row['channel']
        cycle = row['cycle number']
        
        sub = df[(df['channel'] == channel) & (df['cycle number'] == cycle)].copy()
        sub = sub.sort_values('freq/Hz')
        
        real_z = []
        imag_z = []
        
        for freq in freqs:
            freq_data = sub[sub['freq/Hz'] == freq]
            
            re_val = pd.to_numeric(freq_data['Re(Z)/Ohm'], errors='coerce').values[0]
            im_val = pd.to_numeric(freq_data['-Im(Z)/Ohm'], errors='coerce').values[0]
            real_z.append(float(re_val) if not np.isnan(re_val) else np.nan)
            imag_z.append(float(im_val) if not np.isnan(im_val) else np.nan)

        
        state_vectors.append(np.concatenate([real_z, imag_z]))
        sample_ids.append((channel, cycle))

    
    return state_vectors, sample_ids