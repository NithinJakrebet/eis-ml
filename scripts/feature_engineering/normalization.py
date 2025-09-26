"""
Normalization Module

Various normalization and scaling approaches for EIS feature engineering.
"""

import numpy as np

def minmax_normalize(arr):# Min-max normalize an array to [0, 1] range.
    # Compute the minimum and maximum ignoring NaNs
    a_min = np.nanmin(arr)
    a_max = np.nanmax(arr)
    
    # Avoid division by zero: if all values are equal, return zeros
    if a_max != a_min: return (arr - a_min) / (a_max - a_min)
    else: return np.zeros_like(arr)


def standardize(arr): # Z-score standardization (mean=0, std=1).
    mean = np.nanmean(arr)
    std = np.nanstd(arr)
    
    if std != 0: return (arr - mean) / std
    else: return np.zeros_like(arr)


def robust_normalize(arr, q_low=0.25, q_high=0.75): # Robust normalization using quantiles (less sensitive to outliers).
    q1 = np.nanquantile(arr, q_low)
    q3 = np.nanquantile(arr, q_high)
    iqr = q3 - q1
    
    if iqr != 0: return (arr - np.nanmedian(arr)) / iqr
    else: return np.zeros_like(arr)