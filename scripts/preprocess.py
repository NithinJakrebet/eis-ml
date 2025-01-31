import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler

def preprocess_data(df, target_col, feature_cols, sample_frac=0.1):
    """ Drops NaN, samples dataset, and scales features and target. """
    
    df = df.dropna()  # Remove NaN values
    df_sampled = df.sample(frac=sample_frac, random_state=42)  # Take 10% sample
    
    X = df_sampled[feature_cols].values
    y = df_sampled[target_col].values
    
    scaler_X = StandardScaler()
    scaler_y = StandardScaler()
    X_scaled = scaler_X.fit_transform(X)
    y_scaled = scaler_y.fit_transform(y.reshape(-1, 1)).ravel()
    
    return X_scaled, y_scaled, scaler_X, scaler_y
