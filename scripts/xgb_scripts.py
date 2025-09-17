import numpy as np
import xgboost as xgb
import config
import os
import glob

def train_ensemble_model(X_train, y_train, n_models=None, model_params=None):
    if n_models is None: n_models = config.NUM_ENSEMBLE
    if model_params is None: model_params = config.MODEL_PARAMS.copy()
    
    ensemble_models = []
        
    for i in range(n_models):
        model = xgb.XGBRegressor(
            n_estimators=model_params['n_estimators'],
            max_depth=model_params['max_depth'],
            learning_rate=model_params['learning_rate'],
            objective=model_params['objective'],
            random_state=model_params['random_state'] + i
        )
        
        model.fit(X_train, y_train)
        ensemble_models.append(model)
    
    return ensemble_models

def predict_ensemble(models, X_test):
    all_predictions = np.array([model.predict(X_test) for model in models])    
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    
    return mean_predictions, std_predictions, all_predictions

def save_ensemble_models(models, prefix="xgb_ensemble"):
    os.makedirs(config.MODELS_DIR, exist_ok=True)
    
    for i, model in enumerate(models):
        filename = f"{prefix}_model_{i:02d}.pkl"
        filepath = os.path.join(config.MODELS_DIR, filename)
        model.save_model(filepath)
        
    print(f"Saved {len(models)} models to {config.MODELS_DIR}")

def load_ensemble_models(prefix="xgb_ensemble"):
    pattern = os.path.join(config.MODELS_DIR, f"{prefix}_model_*.pkl")
    model_files = sorted(glob.glob(pattern))
    
    if not model_files: raise FileNotFoundError(f"No models found: {pattern}")
    
    models = []
    for filepath in model_files:
        model = xgb.XGBRegressor()
        model.load_model(filepath)
        models.append(model)
    
    print(f"Loaded {len(models)} models from {config.MODELS_DIR}")
    return models
