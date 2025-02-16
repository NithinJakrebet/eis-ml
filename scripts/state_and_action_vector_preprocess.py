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
        action_data_by_cycle,
        expected_eis_dim=162  # or whichever dimension you require
    )
"""

import numpy as np
import pandas as pd


def build_eis_data_by_cycle(df: pd.DataFrame):
    eis_data_by_cycle = {}

    all_cycles = df['cycle number'].unique()

    for cycle in sorted(all_cycles):
        # Subset the dataframe for this cycle
        df_cycle_data = df[df['cycle number'] == cycle].copy()

        # Filter only the EIS rows (e.g. Ns == 1 or 6)
        df_cycle_eis_data = df_cycle_data[df_cycle_data['Ns'].isin([1, 6])].copy()

        # If no EIS data for this cycle, store empty
        if df_cycle_eis_data.empty:
            eis_data_by_cycle[cycle] = {
                "labeled_impedance": {},
                "state_vector": np.array([])
            }
            continue

        # Sort by frequency (ascending)
        df_cycle_eis_data.sort_values(by='freq/Hz', ascending=True, inplace=True)

        # Extract frequency, real part, and imaginary part
        freq_array = df_cycle_eis_data['freq/Hz'].values
        z_re_array = df_cycle_eis_data['Re(Z)/Ohm'].values
        z_im_array = df_cycle_eis_data['Im(Z)/Ohm'].values

        # Build a dictionary that labels each frequency's real/imag parts
        labeled_impedance = {}
        for i, freq_val in enumerate(freq_array):
            freq_str = f"{freq_val:.5g}"
            labeled_impedance[f"Z_re({freq_str}Hz)"] = z_re_array[i]
            labeled_impedance[f"Z_im({freq_str}Hz)"] = z_im_array[i]

        # Create the flattened EIS state vector
        eis_state_vector = np.concatenate([z_re_array, z_im_array])

        # Store in the dictionary
        eis_data_by_cycle[cycle] = {
            "labeled_impedance": labeled_impedance,
            "state_vector": eis_state_vector
        }

    return eis_data_by_cycle

def build_action_data_by_cycle(df: pd.DataFrame):
    action_data_by_cycle = {}
    all_cycles = df['cycle number'].unique()

    for cycle in sorted(all_cycles):
        df_cycle_data = df[df['cycle number'] == cycle].copy()

        # Identify charge vs. discharge
        df_charge_data = df_cycle_data[df_cycle_data['I/mA'] > 0].copy()
        df_discharge_data = df_cycle_data[df_cycle_data['I/mA'] < 0].copy()

        # Compute features for Charge
        i_charge_avg = df_charge_data['I/mA'].mean() if not df_charge_data.empty else 0.0
        total_charge_time = df_charge_data['step time/s'].sum() if not df_charge_data.empty else 0.0

        if not df_charge_data.empty:
            q_charge_start = df_charge_data['Q charge/mA.h'].min()
            q_charge_end   = df_charge_data['Q charge/mA.h'].max()
            q_charge = q_charge_end - q_charge_start
        else:
            q_charge = 0.0

        # Compute features for Discharge
        i_discharge_avg = df_discharge_data['I/mA'].mean() if not df_discharge_data.empty else 0.0
        total_discharge_time = df_discharge_data['step time/s'].sum() if not df_discharge_data.empty else 0.0

        if not df_discharge_data.empty:
            q_discharge_start = df_discharge_data['Q discharge/mA.h'].min()
            q_discharge_end   = df_discharge_data['Q discharge/mA.h'].max()
            q_discharge = q_discharge_end - q_discharge_start
        else:
            q_discharge = 0.0

        # Create the action vector
        usage_action_vector = np.array([
            i_charge_avg,
            total_charge_time,
            q_charge,
            i_discharge_avg,
            total_discharge_time,
            q_discharge
        ], dtype=float)

        action_data_by_cycle[cycle] = usage_action_vector

    return action_data_by_cycle

def build_combined_dataset(
    df: pd.DataFrame,
    eis_data_by_cycle: dict,
    action_data_by_cycle: dict,
):
    X_list = []
    y_list = []

    # Find cycles present in BOTH dictionaries
    common_cycles = sorted(set(eis_data_by_cycle.keys()) & set(action_data_by_cycle.keys()))

    for cycle in common_cycles:
        state_vec = eis_data_by_cycle[cycle]["state_vector"]
        if len(state_vec) != 162:   # expected eis dimensions
            print(f"Skipping cycle {cycle}: EIS vector length {len(state_vec)} != {162}")
            continue

        action_vec = action_data_by_cycle[cycle]
        combined_input = np.concatenate([state_vec, action_vec])

        # Fetch the target capacity Q_n from the original dataframe
        df_cycle = df[df['cycle number'] == cycle]
        if df_cycle.empty:
            print(f"Skipping cycle {cycle}: no rows in df.")
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
