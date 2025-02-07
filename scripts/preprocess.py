# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocesses the dataset by dropping NaN values, filtering relevant rows,
    sampling, and scaling the features and target.

    Args:
        df (pd.DataFrame): The input dataframe.
        target_col (str): The name of the target column.
        feature_cols (list): The list of feature column names.
        sample_frac (float): Fraction of the dataset to sample (default is 0.1).

    Returns:
        tuple: (X_scaled, y_scaled, scaler_X, scaler_y)
    """
    
    # Drop NaN values
    df = df.dropna()
    
    # fix reading column issue
    if "#NAME?" in df.columns.values : df = df.rename(columns={"#NAME?": "Im(Z)/Ohm"})

    # Apply filtering
    # df = df.loc[(df['cycle number'] != 0)].copy()
    # df = df.loc[(df['Ns'].isin([1, 6])) & (df['cycle number'] != 0)].copy()
    
    # Split dataframe by cycle numbers
    # dataframe_dict = {}
    # cycle_nums = df['cycle number'].unique()
    # for cycle in cycle_nums:
    #    dataframe_dict[f'df_cycle_{cycle}'] = df[df['cycle number'] == cycle].copy()
    
    return df
