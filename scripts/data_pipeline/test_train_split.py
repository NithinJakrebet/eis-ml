import pandas as pd
import numpy as np
from .load_single_channel import load_single_channel
import config
import os

def test_train_split(
    data_folder=None, 
    cycle_range=None,
    method="leave_two_out"
):
    if data_folder is None: data_folder = "data/04-03-24"
    
    match method:
        case "leave_two_out": df_train, df_test = leave_two_out(cycle_range, data_folder)
        case "bin_and_split": df_train, df_test = bin_and_split(cycle_range, data_folder)
        case "temporal": df_train, df_test = temporal_split(data_folder)
        case _: raise ValueError(f"Unknown split method: {method}")
        
    return df_train, df_test
   
   
def temporal_split(data_folder):
    all_channels = [f[:-4] for f in os.listdir(data_folder) if f.endswith('.csv')]
    
    all_channels_data = {}
    for channel in all_channels: all_channels_data[channel] = load_single_channel(data_folder, channel, cycle_range=None)
    
    train_dfs = []
    test_dfs = []
    
    for channel, df in all_channels_data.items():
        if df is None or df.empty: continue
        
        all_cycles = sorted(df['cycle number'].unique())
        
        split_index = int(len(all_cycles) * 0.6)
        train_cycles = all_cycles[:split_index]
        test_cycles = all_cycles[split_index:]
        
        train_data = df[df['cycle number'].isin(train_cycles)]
        test_data = df[df['cycle number'].isin(test_cycles)]
        
        if not train_data.empty: train_dfs.append(train_data)
        if not test_data.empty: test_dfs.append(test_data)
    
    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)


            
            
def leave_two_out(cycle_range, data_folder):
    # leave 2 out method
    train_channels = config.TRAIN_CHANNELS
    test_channels = config.TEST_CHANNELS
    train_channels_data = {}
    test_channels_data = {}
    for channel in train_channels: train_channels_data[channel] = load_single_channel(data_folder, channel, cycle_range)    
    for channel in test_channels: test_channels_data[channel] = load_single_channel(data_folder, channel, cycle_range)
    train_dfs = list(train_channels_data.values())
    test_dfs = list(test_channels_data.values())

    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)


   

def bin_and_split(cycle_range, data_folder):
    all_channels = [
        f[:-4] for f in os.listdir(data_folder)
        if f.endswith('.csv')
    ]
    
    all_channels_data = {}
    for channel in all_channels: 
        all_channels_data[channel] = load_single_channel(data_folder, channel, cycle_range)
    
    train_dfs = []
    test_dfs = []
    
    for channel, df in all_channels_data.items():
        if df is None or df.empty: continue
        end_of_cycle_capacities = df.groupby('cycle number')['Capacity/mA.h'].last()
        capacity_bins = pd.qcut(end_of_cycle_capacities, q=10, labels=False, duplicates='drop')
        
        channel_train_data = []
        channel_test_data = []
        
        for bin_num in np.unique(capacity_bins):
            cycles_in_bin = end_of_cycle_capacities[capacity_bins == bin_num].index
            cycles_in_bin = sorted(cycles_in_bin)
            split_point = int(len(cycles_in_bin) * 0.8)
            train_cycles = cycles_in_bin[:split_point]
            test_cycles = cycles_in_bin[split_point:]
            
            for cycle in train_cycles:
                cycle_data = df[df['cycle number'] == cycle]
                channel_train_data.append(cycle_data)
                
            for cycle in test_cycles:
                cycle_data = df[df['cycle number'] == cycle]
                channel_test_data.append(cycle_data)
        
        if channel_train_data: train_dfs.append(pd.concat(channel_train_data, ignore_index=True))
        if channel_test_data: test_dfs.append(pd.concat(channel_test_data, ignore_index=True))
    
    return pd.concat(train_dfs, ignore_index=True), pd.concat(test_dfs, ignore_index=True)
            
       