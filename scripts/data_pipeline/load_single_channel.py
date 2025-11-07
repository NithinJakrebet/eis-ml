import os
import pandas as pd

def load_single_channel(data_folder, channel, cycle_range=None):
    file_path = os.path.join(data_folder, f"{channel}.csv")
    df = pd.read_csv(file_path).dropna()

    if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "-Im(Z)/Ohm"})

    # add provenance
    df["channel"] = channel 

    if cycle_range is not None:
        lo, hi = cycle_range
        df = df[df['cycle number'].between(lo, hi)].copy()
    else:
        max_cycle = df['cycle number'].max()
        df = df[(df['cycle number'] > 0) & (df['cycle number'] < max_cycle)].copy()

    return df
