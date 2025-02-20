"""
This module contains functions to:
1. Build EIS-based 'state' vectors for each cycle.
2. Build usage-based 'action' vectors for each cycle.
3. Combine them into a final (X, y) dataset for ML training.

Example Usage:
    from state_and_action_vector_preprocess import (
        build_eis_data_by_cycle,
        build_action_data_by_cycle,
        build_combined_dataset
    )

    df = pd.read_csv("your_dataset.csv")

    # Step 1: Build EIS-based state vectors
    eis_data_by_cycle = build_eis_data_by_cycle(df)

    # Step 2: Build action vectors
    action_data_by_cycle = build_action_data_by_cycle(df)

    # Step 3: Combine into (X, y)
    X, y = build_combined_dataset(
        df,
        eis_data_by_cycle,
        action_data_by_cycle
    )
"""

import numpy as np
import pandas as pd
from collections import Counter


def build_state_vector(df: pd.DataFrame):
    """
    Processes the dataframe to build an EIS state vector for each cycle.
    The state vector is built using the raw EIS data (Re and Im impedance)
    without any interpolation.
    
    A preprocessing filter is applied to keep only rows where the frequency is >0.2 Hz 
    and <=20,000 Hz.
    """
    # Filter rows to include only frequencies between 0.2 and 20,000 Hz.
    df = df[(df['freq/Hz'] > 0.2) & (df['freq/Hz'] <= 20000)]
    
    eis_data_by_cycle = {}
    
    # Process each cycle separately.
    all_cycles = df['cycle number'].unique()
    
    for cycle in sorted(all_cycles):
        # Filter data for the current cycle.
        df_cycle_data = df[df['cycle number'] == cycle].copy()
        
        # Keep only rows where EIS measurements are taken (Ns=1 or 6).
        df_cycle_eis_data = df_cycle_data[df_cycle_data['Ns'].isin([1, 6])].copy()
        
        # If no EIS data exists for this cycle, store an empty state vector.
        if df_cycle_eis_data.empty:
            eis_data_by_cycle[cycle] = {"labeled_impedance": {}, "state_vector": np.array([])}
            continue
        
        # Sort the data by frequency.
        df_cycle_eis_data.sort_values(by='freq/Hz', ascending=True, inplace=True)
        
        # Extract the frequency and impedance values.
        freq_array = df_cycle_eis_data['freq/Hz'].values
        z_re_array = df_cycle_eis_data['Re(Z)/Ohm'].values
        z_im_array = df_cycle_eis_data['Im(Z)/Ohm'].values
        
        # Normalize the impedance data using Z-score normalization.
        z_re_norm = (z_re_array - np.mean(z_re_array)) / np.std(z_re_array)
        z_im_norm = (z_im_array - np.mean(z_im_array)) / np.std(z_im_array)
        
        # Build a labeled dictionary for debugging/inspection.
        labeled_impedance = {}
        for freq_val, re_val, im_val in zip(freq_array, z_re_norm, z_im_norm):
            freq_str = f"{freq_val:.5g}"
            labeled_impedance[f"Z_re({freq_str}Hz)"] = re_val
            labeled_impedance[f"Z_im({freq_str}Hz)"] = im_val
        
        # Concatenate the normalized real and imaginary parts into one state vector.
        state_vector = np.concatenate([z_re_norm, z_im_norm])
        eis_data_by_cycle[cycle] = {"labeled_impedance": labeled_impedance, "state_vector": state_vector}
    
    return eis_data_by_cycle


def build_action_vector(df: pd.DataFrame):
    """
    Processes the dataframe to build an action vector for each cycle.
    In addition to basic features (average current, total time, net capacity), this
    version also extracts energy features and cycle efficiency.
    """
    action_data_by_cycle = {}
    all_cycles = df['cycle number'].unique()

    for cycle in sorted(all_cycles):
        df_cycle_data = df[df['cycle number'] == cycle].copy()

        # Separate charge and discharge data based on the sign of the current.
        df_charge_data = df_cycle_data[df_cycle_data['I/mA'] > 0].copy()
        df_discharge_data = df_cycle_data[df_cycle_data['I/mA'] < 0].copy()

        # ---- Charging Features ----
        if not df_charge_data.empty:
            i_charge_avg = df_charge_data['I/mA'].mean()
            total_charge_time = df_charge_data['step time/s'].sum()
            q_charge = df_charge_data['Q charge/mA.h'].max() - df_charge_data['Q charge/mA.h'].min()
            energy_charge = df_charge_data['Energy charge/W.h'].sum()
        else:
            i_charge_avg = total_charge_time = q_charge = energy_charge = 0.0

        # ---- Discharging Features ----
        if not df_discharge_data.empty:
            i_discharge_avg = df_discharge_data['I/mA'].mean()
            total_discharge_time = df_discharge_data['step time/s'].sum()
            q_discharge = df_discharge_data['Q discharge/mA.h'].max() - df_discharge_data['Q discharge/mA.h'].min()
            energy_discharge = df_discharge_data['Energy discharge/W.h'].sum()
        else:
            i_discharge_avg = total_discharge_time = q_discharge = energy_discharge = 0.0

        # ---- Overall Cycle Efficiency ----
        efficiency_avg = df_cycle_data['Efficiency/%'].mean() if not df_cycle_data.empty else 0.0

        # ---- Construct Enhanced Action Vector ----
        # Features: [avg_charge_current, total_charge_time, net_charge, energy_charge,
        #            avg_discharge_current, total_discharge_time, net_discharge, energy_discharge, efficiency]
        usage_action_vector = np.array([
            i_charge_avg,
            total_charge_time,
            q_charge,
            energy_charge,
            i_discharge_avg,
            total_discharge_time,
            q_discharge,
            energy_discharge,
            efficiency_avg
        ], dtype=float)

        action_data_by_cycle[cycle] = usage_action_vector

    return action_data_by_cycle


def get_mode_unique_frequency_count(eis_data_by_cycle: dict) -> int:
    """
    Determines the most common (mode) number of unique frequency points across cycles.
    Each state's vector length is assumed to be twice the number of unique frequencies
    (concatenated Re and Im parts).
    
    Returns:
        mode_count (int): The most common number of unique frequency points.
    """
    unique_counts = []
    for cycle, data in eis_data_by_cycle.items():
        if data["state_vector"].size > 0:
            freq_count = len(data["state_vector"]) // 2
            unique_counts.append(freq_count)
    if not unique_counts:
        raise ValueError("No valid state vectors found in eis_data_by_cycle!")
    
    mode_count = Counter(unique_counts).most_common(1)[0][0]
    return mode_count


def combine_vector(df: pd.DataFrame, eis_data_by_cycle: dict, action_data_by_cycle: dict):
    """
    Combines the state (EIS) and action vectors into a single feature vector per cycle.
    The target is taken as the maximum discharge capacity (Q discharge) for each cycle.

    This function determines the most common number of unique frequency measurements (mode)
    using the helper method `get_mode_unique_frequency_count`, and only uses cycles that
    contain that many unique frequency points.
    """
    X_list = []
    y_list = []
    
    # Get the most common number of unique frequency points.
    mode_count = get_mode_unique_frequency_count(eis_data_by_cycle)
    expected_state_vec_length = mode_count * 2
    print(f"Most common unique frequency count is {mode_count}, so expected EIS state vector length is {expected_state_vec_length}.")
    
    # Consider only cycles present in both dictionaries.
    common_cycles = sorted(set(eis_data_by_cycle.keys()) & set(action_data_by_cycle.keys()))
    
    for cycle in common_cycles:
        state_vec = eis_data_by_cycle[cycle]["state_vector"]
        # Check if this cycle has the expected number of unique frequency points.
        if state_vec.size == 0 or (len(state_vec) // 2) != mode_count:
            print(f"Skipping cycle {cycle}: has {(len(state_vec) // 2) if state_vec.size > 0 else 0} unique frequency points, expected {mode_count}.")
            continue
        
        action_vec = action_data_by_cycle[cycle]
        # Concatenate the state vector (EIS features) and the action vector (cycling protocol features).
        combined_input = np.concatenate([state_vec, action_vec])
        
        # The target y is defined as the maximum discharge capacity for that cycle.
        df_cycle = df[df['cycle number'] == cycle]
        if df_cycle.empty:
            print(f"Skipping cycle {cycle}: no data in df.")
            continue
        Q_n = df_cycle['Q discharge/mA.h'].max()
        if pd.isna(Q_n):
            print(f"Skipping cycle {cycle}: Q_n is NaN.")
            continue
        
        X_list.append(combined_input)
        y_list.append(Q_n)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print("Final dataset shapes:")
    print("  X:", X.shape)
    print("  y:", y.shape)
    return X, y


def preprocess_SAV(df: pd.DataFrame):
    # Build EIS data, action data, then combine to (X, y)
    eis_data_by_cycle = build_state_vector(df)
    action_data_by_cycle = build_action_vector(df)
    X, y = combine_vector(df, eis_data_by_cycle, action_data_by_cycle)

    print("X shape:", X.shape)
    print("y shape:", y.shape)
    
    return X, y