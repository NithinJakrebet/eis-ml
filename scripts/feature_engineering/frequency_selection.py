"""
Frequency Selection Module

Physics-informed frequency selection strategies for EIS analysis.
"""

import numpy as np
import pandas as pd


def select_frequencies(df: pd.DataFrame, n_frequencies=3):
    """
    Apply physics-informed frequency selection.
    
    Args:
        df: DataFrame with EIS measurements
        n_frequencies: Number of frequencies to select
        
    Returns:
        List of selected frequencies sorted in ascending order
    """
    frequencies = np.array(sorted(df['freq/Hz'].unique()))
    physics_mask = (frequencies >= 1.0) & (frequencies <= 10.0)
    selected_freqs = frequencies[physics_mask]
    
    if len(selected_freqs) < n_frequencies:
        # Add closest frequencies if not enough in range
        remaining = n_frequencies - len(selected_freqs)
        other_freqs = frequencies[~physics_mask]
        distances = np.abs(other_freqs - 5.0)  # Distance from 5 Hz
        closest_indices = np.argsort(distances)[:remaining]
        selected_freqs = np.concatenate([selected_freqs, other_freqs[closest_indices]])
    else:
        # If we have more than needed, prioritize frequencies closest to data-driven optimal: 3.2, 5.6 Hz
        if len(selected_freqs) > n_frequencies:
            optimal_targets = [3.2, 5.6, 10.0, 2.4, 1.8]  # Based on stability analysis
            
            # Calculate distances to optimal frequencies and select closest
            scores = []
            for freq in selected_freqs:
                min_distance = min(abs(freq - target) for target in optimal_targets)
                scores.append(min_distance)
            
            # Select frequencies with smallest distances to optimal targets
            best_indices = np.argsort(scores)[:n_frequencies]
            selected_freqs = selected_freqs[best_indices]
            
    print(f"Selected frequencies: {sorted(selected_freqs)}")
    return sorted(selected_freqs)


def get_physics_frequencies():
    """Get the standard physics-informed frequency range."""
    return [1.78, 5.62, 10.0]  # Default physics-based selection


def get_optimal_frequencies():
    """Get data-driven optimal frequencies from stability analysis."""
    return [3.2, 5.6, 10.0, 2.4, 1.8]