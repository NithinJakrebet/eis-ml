import numpy as np
import pandas as pd

def build_state_vector(
    df: pd.DataFrame, 
    selected_frequencies=None,
    components_to_use=None
    ): 
    df = (
        df.dropna()
          .query("`freq/Hz` > 0.2 and `freq/Hz` <= 20000")
          .query("Ns in [1]")
    )
    
    # Verify channel column exists
    if 'channel' not in df.columns: 
        raise ValueError("DataFrame must have a 'channel' column. Use load_single_channel to add it.")
    
    frequencies_to_use = selected_frequencies if selected_frequencies is not None else sorted(df['freq/Hz'].unique())
    
    channel_cycle_pairs = df.groupby(['channel', 'cycle number'], as_index=False).first()[['channel', 'cycle number']]
    
    state_vectors = []
    sample_ids = []
    
    for _, row in channel_cycle_pairs.iterrows():
        channel = row['channel']
        cycle = row['cycle number']
        
        # Extract data for this specific (channel, cycle) pair
        sub = df[(df['channel'] == channel) & (df['cycle number'] == cycle)].copy()
        sub = sub.sort_values('freq/Hz')
        
        real_z = []
        imag_z = []
        
        for freq in frequencies_to_use:
            freq_data = sub[sub['freq/Hz'] == freq]
            if not freq_data.empty:
                # Use median aggregation for robustness against duplicates
                re_val = pd.to_numeric(freq_data['Re(Z)/Ohm'], errors='coerce').median()
                im_val = pd.to_numeric(freq_data['Im(Z)/Ohm'], errors='coerce').median()
                real_z.append(float(re_val) if not np.isnan(re_val) else np.nan)
                imag_z.append(float(im_val) if not np.isnan(im_val) else np.nan)
            else:
                real_z.append(np.nan)
                imag_z.append(np.nan)
        
        # Build feature vector based on component selection
        if components_to_use is not None:
            selected_features = []
            for i, component_code in enumerate(components_to_use):
                if component_code == 0:  
                    selected_features.append(real_z[i])
                elif component_code == 1: 
                    selected_features.append(imag_z[i])
            state_vectors.append(np.array(selected_features))
        else: 
            state_vectors.append(np.concatenate([real_z, imag_z]))
        
        sample_ids.append((channel, cycle))
    
    return state_vectors, sample_ids