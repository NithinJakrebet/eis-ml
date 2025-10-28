import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Model parameters - simple, effective defaults
MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 100,
    'learning_rate': 0.1,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 50  # Moved here for older XGBoost versions
}


def train_ensemble_model(X_train, y_train, n_models=5):
    """
    Trains an ensemble of XGBoost models.

    Args:
        X_train (np.array): Training feature data.
        y_train (np.array): Training target data.
        n_models (int): The number of models in the ensemble.

    Returns:
        list: A list of trained XGBoost model objects.
    """
    models = []
    print(f"Training an ensemble of {n_models} models...")
    for i in range(n_models):
        # Use a different random seed for each model to ensure diversity
        random_seed = 42 + i
        
        # Create a different train/validation split for each model's early stopping
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_seed
        )
        
        # Update model params with the new random state
        params = MODEL_PARAMS.copy()
        params['random_state'] = random_seed
        
        model = xgb.XGBRegressor(**params)
        model.fit(
            X_tr, y_tr,
            eval_set=[(X_val, y_val)],
            verbose=False
        )
        models.append(model)
        
    return models

def predict_ensemble(models, X_test):
    all_predictions = np.array([model.predict(X_test) for model in models])
    
    # Calculate the mean and standard deviation of the predictions
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    
    return mean_predictions, std_predictions, all_predictions