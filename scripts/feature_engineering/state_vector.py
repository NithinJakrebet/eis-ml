import numpy as np
import pandas as pd

def build_state_vector(
    df: pd.DataFrame,
    frequencies: list = None
):
    if 'channel' not in df.columns: raise ValueError("DataFrame must have a 'channel' column. Use load_single_channel to add it.")
    
    ns_states = [1, 6]
    freq_range = (0.2, 20000)
    
    df_filtered = (
        df.dropna()
          .query(f"`freq/Hz` > {freq_range[0]} and `freq/Hz` <= {freq_range[1]}")
          .query(f"Ns in {ns_states}")
    )
    
    ns_frequencies = {}
    for ns in ns_states:
        ns_data = df_filtered[df_filtered['Ns'] == ns]
        ns_frequencies[ns] = sorted(ns_data['freq/Hz'].unique())
        print(f"[Ns={ns}] Found {len(ns_frequencies[ns])} unique frequencies")
    
    channel_cycle_pairs = []
    for (channel, cycle), group in df_filtered.groupby(['channel', 'cycle number']):
        ns_present = set(group['Ns'].unique())
        if all(ns in ns_present for ns in ns_states):
            channel_cycle_pairs.append((channel, cycle))
    
    print(f"Found {len(channel_cycle_pairs)} complete cycles with all Ns states {ns_states}")
    
    state_vectors = []
    sample_ids = []
    
    for channel, cycle in channel_cycle_pairs:
        cycle_features = []
        
        for ns in ns_states:
            sub = df_filtered[
                (df_filtered['channel'] == channel) & 
                (df_filtered['cycle number'] == cycle) & 
                (df_filtered['Ns'] == ns)
            ].copy().sort_values('freq/Hz')
            
            real_z = []
            imag_z = []
            
            for freq in ns_frequencies[ns]:
                freq_data = sub[sub['freq/Hz'] == freq]
                
                if len(freq_data) > 0:
                    re_val = pd.to_numeric(freq_data['Re(Z)/Ohm'], errors='coerce').values[0]
                    im_val = pd.to_numeric(freq_data['-Im(Z)/Ohm'], errors='coerce').values[0]
                    real_z.append(float(re_val) if not np.isnan(re_val) else np.nan)
                    imag_z.append(float(im_val) if not np.isnan(im_val) else np.nan)
                else:
                    real_z.append(np.nan)
                    imag_z.append(np.nan)
            
            cycle_features.extend(real_z)
            cycle_features.extend(imag_z)
        
        state_vectors.append(cycle_features)
        sample_ids.append((channel, cycle))
    
    S = np.asarray(state_vectors, dtype=float)
    print(f"State vector shape: {S.shape} (samples x features)")
    
    return S, sample_ids