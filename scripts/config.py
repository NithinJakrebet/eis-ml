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
    'n_estimators': 300,
    'max_depth': 6,
    'learning_rate': 0.1,
    'objective': 'reg:squarederror',
    'random_state': 42
}

# Ensemble parameters (CV folds)
NUM_ENSEMBLE = 10

# Results configuration
RESULTS_DIR = "../results"
MODELS_DIR = "../models"