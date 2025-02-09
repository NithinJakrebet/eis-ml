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


def build_eis_data_by_cycle(
    df: pd.DataFrame,
    freq_col: str = 'freq/Hz',
    re_col: str = 'Re(Z)/Ohm',
    im_col: str = 'Im(Z)/Ohm',
    cycle_col: str = 'cycle number',
    ns_col: str = 'Ns',
    eis_states = [1, 6]
):
    """
    Build EIS-based state vectors from the given dataframe.

    Args:
        df (pd.DataFrame): The input DataFrame containing EIS data.
        freq_col (str): The column name for frequency.
        re_col (str): The column name for the real part of impedance.
        im_col (str): The column name for the imaginary part of impedance.
        cycle_col (str): The column identifying cycles (e.g. 'cycle number').
        ns_col (str): The column identifying step numbers/states (e.g. 'Ns').
        eis_states (List[int]): The step indices (Ns) corresponding to EIS measurements.

    Returns:
        dict: A dictionary keyed by cycle, where each value is another dict:
              {
                  "labeled_impedance": dict of { "Z_re(freqHz)": val, "Z_im(freqHz)": val, ... },
                  "state_vector": np.ndarray (the concatenated real+imag EIS data)
              }
    """
    eis_data_by_cycle = {}

    all_cycles = df[cycle_col].unique()

    for cycle in sorted(all_cycles):
        # Subset the dataframe for this cycle
        df_cycle_data = df[df[cycle_col] == cycle].copy()

        # Filter only the EIS rows (e.g. Ns == 1 or 6)
        df_cycle_eis_data = df_cycle_data[df_cycle_data[ns_col].isin(eis_states)].copy()

        # If no EIS data for this cycle, store empty
        if df_cycle_eis_data.empty:
            eis_data_by_cycle[cycle] = {
                "labeled_impedance": {},
                "state_vector": np.array([])
            }
            continue

        # Sort by frequency (ascending)
        df_cycle_eis_data.sort_values(by=freq_col, ascending=True, inplace=True)

        # Extract frequency, real part, and imaginary part
        freq_array = df_cycle_eis_data[freq_col].values
        z_re_array = df_cycle_eis_data[re_col].values
        z_im_array = df_cycle_eis_data[im_col].values

        # Build a dictionary that labels each frequency's real/imag parts
        labeled_impedance = {}
        for i, freq_val in enumerate(freq_array):
            freq_str = f"{freq_val:.5g}"  # e.g. "0.01", "100", etc.
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


def build_action_data_by_cycle(
    df: pd.DataFrame,
    cycle_col: str = 'cycle number',
    current_col: str = 'I/mA',
    step_time_col: str = 'step time/s',
    q_charge_col: str = 'Q charge/mA.h',
    q_discharge_col: str = 'Q discharge/mA.h'
):
    """
    Build action (usage) vectors from the given dataframe.
    Typically, we treat positive I/mA as charge and negative as discharge.

    Args:
        df (pd.DataFrame): The input DataFrame.
        cycle_col (str): Column with cycle identifiers.
        current_col (str): Column with measured current (mA).
        step_time_col (str): Column with step time (s).
        q_charge_col (str): Column with cumulative charge (mA.h).
        q_discharge_col (str): Column with cumulative discharge (mA.h).

    Returns:
        dict: A dictionary keyed by cycle, each containing a 6-element np.array:
              [i_charge_avg, total_charge_time, q_charge,
               i_discharge_avg, total_discharge_time, q_discharge]
    """
    action_data_by_cycle = {}
    all_cycles = df[cycle_col].unique()

    for cycle in sorted(all_cycles):
        df_cycle_data = df[df[cycle_col] == cycle].copy()

        # Identify charge vs. discharge
        df_charge_data = df_cycle_data[df_cycle_data[current_col] > 0].copy()
        df_discharge_data = df_cycle_data[df_cycle_data[current_col] < 0].copy()

        # Compute features for Charge
        i_charge_avg = df_charge_data[current_col].mean() if not df_charge_data.empty else 0.0
        total_charge_time = df_charge_data[step_time_col].sum() if not df_charge_data.empty else 0.0

        if not df_charge_data.empty:
            q_charge_start = df_charge_data[q_charge_col].min()
            q_charge_end   = df_charge_data[q_charge_col].max()
            q_charge = q_charge_end - q_charge_start
        else:
            q_charge = 0.0

        # Compute features for Discharge
        i_discharge_avg = df_discharge_data[current_col].mean() if not df_discharge_data.empty else 0.0
        total_discharge_time = df_discharge_data[step_time_col].sum() if not df_discharge_data.empty else 0.0

        if not df_discharge_data.empty:
            q_discharge_start = df_discharge_data[q_discharge_col].min()
            q_discharge_end   = df_discharge_data[q_discharge_col].max()
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
    cycle_col: str = 'cycle number',
    q_discharge_col: str = 'Q discharge/mA.h',
    expected_eis_dim: int = 162,
):
    """
    Combine EIS-based state vectors and usage-based action vectors to form final (X, y) arrays.

    Args:
        df (pd.DataFrame): Original dataframe for reference to fetch targets (capacity).
        eis_data_by_cycle (dict): Output from build_eis_data_by_cycle().
        action_data_by_cycle (dict): Output from build_action_data_by_cycle().
        cycle_col (str): Column identifying cycles.
        q_discharge_col (str): Column for final discharge capacity used as target.
        expected_eis_dim (int): Expected length of the EIS vector (skip cycles not matching).

    Returns:
        X (np.ndarray): 2D array of shape (num_valid_cycles, dim_state + dim_action).
        y (np.ndarray): 1D array of discharge capacities for each cycle.
    """
    X_list = []
    y_list = []

    # Find cycles present in BOTH dictionaries
    common_cycles = sorted(set(eis_data_by_cycle.keys()) & set(action_data_by_cycle.keys()))

    for cycle in common_cycles:
        state_vec = eis_data_by_cycle[cycle]["state_vector"]
        if len(state_vec) != expected_eis_dim:
            print(f"Skipping cycle {cycle}: EIS vector length {len(state_vec)} != {expected_eis_dim}")
            continue

        action_vec = action_data_by_cycle[cycle]
        combined_input = np.concatenate([state_vec, action_vec])

        # Fetch the target capacity Q_n from the original dataframe
        df_cycle = df[df[cycle_col] == cycle]
        if df_cycle.empty:
            print(f"Skipping cycle {cycle}: no rows in df.")
            continue

        Q_n = df_cycle[q_discharge_col].max()
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
