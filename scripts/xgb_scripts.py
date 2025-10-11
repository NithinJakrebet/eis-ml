import numpy as np
import xgboost as xgb
import config

def train_ensemble_model(X_train, y_train):
    from sklearn.model_selection import KFold
    
    n_folds = config.NUM_ENSEMBLE
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=42)
    ensemble_models = []
    
    for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X_train)):
        X_fold_train = X_train[train_idx]
        y_fold_train = y_train[train_idx]
        
        model_params = config.MODEL_PARAMS.copy()
        model_params['random_state'] = model_params.get('random_state', 42) + fold_idx
        
        model = xgb.XGBRegressor(**model_params)
        model.fit(X_fold_train, y_fold_train)
        ensemble_models.append(model)
    
    return ensemble_models

def predict_ensemble(models, X_test):
    all_predictions = np.array([model.predict(X_test) for model in models])    
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    return mean_predictions, std_predictions, all_predictions