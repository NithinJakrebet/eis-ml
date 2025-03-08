import numpy as np
import pandas as pd
from collections import Counter
import os

def load_data(subfolder: str, filename: str, data_dir: str = "../data") -> pd.DataFrame:
    df = pd.read_csv(os.path.join(data_dir, subfolder, filename))
    df = df.dropna()

    # Fix column name issue
    if "#NAME?" in df.columns: df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})
        
    # drop the 0th cycle
    df = df.loc[(df['cycle number'] != 0)].copy()

    return df

def build_state_vector_binned(df: pd.DataFrame, n_bins: int = 50):
    """
    Builds an EIS state vector for each cycle by binning the frequency data.
    
    For each cycle, 
        filter to frequencies between 0.2 and 20,000 Hz,
        force key columns to be numeric, 
        and then divide the frequency range into n_bins logarithmically spaced bins.
        For each bin, compute the mean and standard deviation of both Re and Im. 
        The state vector is the concatenation of these 4 features per bin.
    """
    for col in ['freq/Hz', 'Re(Z)/Ohm', 'Im(Z)/Ohm', 'Ns']:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    df = df.dropna(subset=['freq/Hz', 'Re(Z)/Ohm', 'Im(Z)/Ohm', 'Ns'])
    df = df[(df['freq/Hz'] > 0.2) & (df['freq/Hz'] <= 20000)]
    
    cycles = []
    state_vectors = []
    
    for cycle in sorted(df['cycle number'].unique()):
        df_cycle = df[df['cycle number'] == cycle].copy()
        df_eis = df_cycle[df_cycle['Ns'].isin([1, 6])].copy()
        if df_eis.empty:
            continue  # Skip cycles with no EIS data.
        
        df_eis.sort_values(by='freq/Hz', inplace=True)
        freq = df_eis['freq/Hz'].values
        z_re = df_eis['Re(Z)/Ohm'].values
        z_im = df_eis['Im(Z)/Ohm'].values
        
        bins = np.logspace(np.log10(0.2), np.log10(20000), n_bins + 1)
        bin_indices = np.digitize(freq, bins)
        
        bin_features = []
        for b in range(1, n_bins + 1):
            mask = (bin_indices == b)
            if np.any(mask):
                mean_re = np.mean(z_re[mask])
                std_re = np.std(z_re[mask])
                mean_im = np.mean(z_im[mask])
                std_im = np.std(z_im[mask])
            else:
                mean_re = std_re = mean_im = std_im = 0.0
            bin_features.extend([mean_re, std_re, mean_im, std_im])
        
        state_vector = np.nan_to_num(np.array(bin_features), nan=0.0)
        cycles.append(cycle)
        state_vectors.append(state_vector)
    
    return np.array(cycles), np.array(state_vectors)


def build_action_vector(df: pd.DataFrame):
    """
    Builds an action vector for each cycle based on cycling features.
    
    The action vector includes:
      - Average charging current,
      - Total charging time,
      - Net charge,
      - Energy during charge,
      - Average discharging current,
      - Total discharging time,
      - Net discharge,
      - Energy during discharge,
      - Cycle efficiency.
    
    Returns:
        cycles: a NumPy array of cycle numbers
        action_vectors: a NumPy array where each row is the action vector for that cycle
    """
    cycles = []
    action_vectors = []
    
    for cycle in sorted(df['cycle number'].unique()):
        df_cycle = df[df['cycle number'] == cycle].copy()
        df_charge = df_cycle[df_cycle['I/mA'] > 0].copy()
        df_discharge = df_cycle[df_cycle['I/mA'] < 0].copy()
        
        if not df_charge.empty:
            i_charge_avg = df_charge['I/mA'].mean()
            total_charge_time = df_charge['step time/s'].sum()
            q_charge = df_charge['Q charge/mA.h'].max() - df_charge['Q charge/mA.h'].min()
            energy_charge = df_charge['Energy charge/W.h'].sum()
        else:
            i_charge_avg = total_charge_time = q_charge = energy_charge = 0.0
        
        if not df_discharge.empty:
            i_discharge_avg = df_discharge['I/mA'].mean()
            total_discharge_time = df_discharge['step time/s'].sum()
            q_discharge = df_discharge['Q discharge/mA.h'].max() - df_discharge['Q discharge/mA.h'].min()
            energy_discharge = df_discharge['Energy discharge/W.h'].sum()
        else:
            i_discharge_avg = total_discharge_time = q_discharge = energy_discharge = 0.0
        
        efficiency_avg = df_cycle['Efficiency/%'].mean() if not df_cycle.empty else 0.0
        
        action_vector = np.array([
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
        
        cycles.append(cycle)
        action_vectors.append(action_vector)
    
    return np.array(cycles), np.array(action_vectors)


def combine_vector(df: pd.DataFrame, cycles_state, state_vectors, cycles_action, action_vectors):
    """
    Combines the state and action vectors for cycles that have both,
    and creates the input (X) and target (y) vectors.
    
    The target is the maximum Q discharge value for each cycle.
    """
    common_cycles = np.intersect1d(cycles_state, cycles_action)
    X_list = []
    y_list = []
    
    for cycle in common_cycles:
        idx_state = np.where(cycles_state == cycle)[0][0]
        idx_action = np.where(cycles_action == cycle)[0][0]
        state_vec = state_vectors[idx_state]
        action_vec = action_vectors[idx_action]
        combined = np.concatenate([state_vec, action_vec])
        
        df_cycle = df[df['cycle number'] == cycle]
        if df_cycle.empty:
            continue
        Q_n = df_cycle['Q discharge/mA.h'].max()
        if pd.isna(Q_n):
            continue
        
        X_list.append(combined)
        y_list.append(Q_n)
    
    X = np.array(X_list)
    y = np.array(y_list)
    
    print("Final dataset shapes:")
    print("  X:", X.shape)
    print("  y:", y.shape)
    return X, y


def preprocess_SAV(df: pd.DataFrame, n_bins: int = 50):
    """
    Preprocesses the DataFrame to generate (X, y) for machine learning.
    
    It builds the state vector using binned EIS data (with n_bins bins) and the action vector,
    then combines them.
    """
    df = df.copy()
    for col in df.columns:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    cycles_state, state_vectors = build_state_vector_binned(df, n_bins=n_bins)
    cycles_action, action_vectors = build_action_vector(df)
    X, y = combine_vector(df, cycles_state, state_vectors, cycles_action, action_vectors)
    
    print("X shape:", X.shape)
    print("y shape:", y.shape)
    return X, y
