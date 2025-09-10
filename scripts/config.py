"""
Configuration file for EIS-ML project
"""

# Data configuration
CHANNELS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
TRAIN_CHANNELS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6']
TEST_CHANNELS = ['A7', 'A8']

# EIS data filtering parameters
FREQ_MIN = 0.2
FREQ_MAX = 20000
NS_VALUES = [1, 6]

# Feature engineering parameters
N_FEATURES = 100  # Number of top features to extract per impedance type

# Model parameters
MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 100,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# Ensemble parameters
NUM_ENSEMBLE = 10

# Data folder
DEFAULT_DATA_FOLDER = "03-06-24"

# Results configuration
RESULTS_DIR = "../results"
MODELS_DIR = "../models"

def print_config():
    print("=== EIS-ML Configuration ===")
    print(f"Data folder: {DEFAULT_DATA_FOLDER}")
    print(f"Training channels: {TRAIN_CHANNELS}")
    print(f"Testing channels: {TEST_CHANNELS}")
    print(f"Features: {N_FEATURES}")
    print(f"Ensemble size: {NUM_ENSEMBLE}")