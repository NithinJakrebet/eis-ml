# Data configuration
CHANNELS = ['A1', 'A2', 'A3', 'A4', 'A5', 'A6', 'A7', 'A8']
TRAIN_CHANNELS = ['A1', 'A2', 'A3', 'A4', 'A7', 'A8']
TEST_CHANNELS = ['A2', 'A5']

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

# Results configuration
RESULTS_DIR = "../results"
MODELS_DIR = "../models"


# config.py (add these if you want to override defaults)
GPR_PARAMS = {
    "noise_level": 1e-6,
    "length_scale_bounds": (1e-5, 1e5),
    "noise_level_bounds": (1e-12, 1e-3),
    "normalize_y": True,
    "n_restarts_optimizer": 5,
    "random_state": 42,
}