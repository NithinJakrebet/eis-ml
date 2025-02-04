# preprocess.py
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df: pd.DataFrame, target_col: str, feature_cols: list):
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
    df = df.loc[(df['Ns'].isin([1, 6])) & (df['cycle number'] != 0)].copy()

    # Extract features and target
    X = df[feature_cols].values
    y = df[target_col].values

    # Scale features and target
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()

    return X_scaled, y_scaled, scaler_X, scaler_y
