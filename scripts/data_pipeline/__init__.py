"""
Data Pipeline Package

This package provides modular data loading, preprocessing, and train/test splitting functionality
for EIS (Electrochemical Impedance Spectroscopy) battery degradation analysis.

Main modules:
- main.py: Primary data loading and preparation functions
- cell_analysis.py: Cell capacity analysis and degradation assessment
- cell_disjoint.py: Cell-disjoint splitting strategy (Nature Communications approach)
- stratified_fold.py: Enhanced stratified splitting with automatic binning
- original_splits.py: Original splitting methods for backwards compatibility
"""

# Import main functions for easy access
from .main import load_and_prepare_data
from .cell_analysis import analyze_cell_capacity_ranges
from .cell_disjoint import create_cell_disjoint_split
from .stratified_fold import create_improved_stratified_split

__all__ = [
    'load_and_prepare_data',
    'analyze_cell_capacity_ranges', 
    'create_cell_disjoint_split',
    'create_improved_stratified_split'
]