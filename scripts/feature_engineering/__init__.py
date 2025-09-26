"""
Feature Engineering Package

This package provides modular feature extraction and preprocessing functionality
for EIS (Electrochemical Impedance Spectroscopy) battery degradation analysis.

Main modules:
- main.py: Primary feature building and preprocessing functions
- state_vector.py: State vector construction from impedance data
- action_vector.py: Action vector construction from cycle data
- frequency_selection.py: Physics-informed frequency selection strategies
- normalization.py: Various normalization and scaling approaches
"""

# Import main functions for easy access
from .main import build_model_input
from .state_vector import build_state_vector
from .action_vector import build_action_vector
from .normalization import minmax_normalize

__all__ = [
    'build_model_input',
    'build_state_vector', 
    'build_action_vector',
    'minmax_normalize'
]