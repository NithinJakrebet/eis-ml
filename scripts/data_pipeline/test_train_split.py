import pandas as pd
from .load_single_channel import load_single_channel
import config

def test_train_split(
    data_folder=None, 
    cycle_range=None,
):
    if data_folder is None: data_folder = "data/04-03-24"
    
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



'''
could split based on capacity ranges instead of channels

'''

