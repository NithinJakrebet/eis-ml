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
    'early_stopping_rounds': 50  
}


def train_ensemble_model(X_train, y_train, n_models=5):
    models = []
    print(f"Training an ensemble of {n_models} models...")
    for i in range(n_models):
        random_seed = 42 + i
        
        X_tr, X_val, y_tr, y_val = train_test_split(
            X_train, y_train, test_size=0.2, random_state=random_seed)
        
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
    
    mean_predictions = np.mean(all_predictions, axis=0)
    std_predictions = np.std(all_predictions, axis=0)
    
    return mean_predictions, std_predictions, all_predictions