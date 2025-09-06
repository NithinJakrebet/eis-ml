"""
Model training utilities for EIS-ML project
"""

import numpy as np
import xgboost as xgb
import config

def train_ensemble_model(X_train, y_train, n_models=None, model_params=None):
    """
    Train an ensemble of XGBoost models.
    
    Args:
        X_train (np.ndarray): Training features
        y_train (np.ndarray): Training targets
        n_models (int): Number of models in ensemble. If None, uses config.NUM_ENSEMBLE
        model_params (dict): Model parameters. If None, uses config.MODEL_PARAMS
        
    Returns:
        list: List of trained XGBoost models
    """
    if n_models is None:
        n_models = config.NUM_ENSEMBLE
    if model_params is None:
        model_params = config.MODEL_PARAMS.copy()
    
    ensemble_models = []
    
    print(f"Training ensemble of {n_models} XGBoost models...")
    
    for i in range(n_models):
        # Create model with unique random state
        model = xgb.XGBRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            learning_rate=model_params['learning_rate'],
            objective=model_params['objective'],
            random_state=model_params['random_state'] + i
        )
        
        # Train model
        model.fit(X_train, y_train)
        ensemble_models.append(model)
        
        if (i + 1) % 2 == 0 or i == 0:
            print(f"  Trained model {i + 1}/{n_models}")
    
    print("Ensemble training complete!")
    return ensemble_models

def predict_ensemble(models, X_test):
    """
    Make predictions using an ensemble of models.
    
    Args:
        models (list): List of trained models
        X_test (np.ndarray): Test features
        
    Returns:
        tuple: (mean_predictions, std_predictions, all_predictions)
    """
    # Get predictions from each model
    all_predictions = np.array([model.predict(X_test) for model in models])
    
    # Calculate ensemble statistics
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    
    return mean_predictions, std_predictions, all_predictions

def save_ensemble_models(models, model_dir=None, prefix="xgb_ensemble"):
    """
    Save ensemble models to disk.
    
    Args:
        models (list): List of trained models
        model_dir (str): Directory to save models. If None, uses config.MODELS_DIR
        prefix (str): Prefix for model filenames
    """
    if model_dir is None:
        model_dir = config.MODELS_DIR
    
    import os
    os.makedirs(model_dir, exist_ok=True)
    
    for i, model in enumerate(models):
        filename = f"{prefix}_model_{i:02d}.pkl"
        filepath = os.path.join(model_dir, filename)
        model.save_model(filepath)
        
    print(f"Saved {len(models)} models to {model_dir}")

def load_ensemble_models(model_dir=None, prefix="xgb_ensemble"):
    """
    Load ensemble models from disk.
    
    Args:
        model_dir (str): Directory containing models. If None, uses config.MODELS_DIR
        prefix (str): Prefix for model filenames
        
    Returns:
        list: List of loaded models
    """
    if model_dir is None:
        model_dir = config.MODELS_DIR
    
    import os
    import glob
    
    pattern = os.path.join(model_dir, f"{prefix}_model_*.pkl")
    model_files = sorted(glob.glob(pattern))
    
    if not model_files:
        raise FileNotFoundError(f"No models found matching pattern: {pattern}")
    
    models = []
    for filepath in model_files:
        model = xgb.XGBRegressor()
        model.load_model(filepath)
        models.append(model)
    
    print(f"Loaded {len(models)} models from {model_dir}")
    return models
