import numpy as np
import pandas as pd

def get_physics_frequencies():
    """Get the standard physics-informed frequency range."""
    return [1.78, 5.62, 10.0]  # Default physics-based selection


def get_optimal_frequencies():
    """Get data-driven optimal frequencies from stability analysis."""
    return [3.2, 5.6, 10.0, 2.4, 1.8]