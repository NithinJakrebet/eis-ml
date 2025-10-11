import pandas as pd
import os

def load_single_channel(data_folder, channel, cycle_range=None):
    file_path = os.path.join(data_folder, f"{channel}.csv")
    df = pd.read_csv(file_path)
    df = df.dropna()
    
    if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})

    if cycle_range is not None:
        start_cycle, end_cycle = cycle_range
        df = df[(df['cycle number'] >= start_cycle) & (df['cycle number'] <= end_cycle)].copy()
    else:
        # Load all cycles except the last one
        max_cycle = df['cycle number'].max()
        df = df[df['cycle number'] < max_cycle].copy()
    
    return df