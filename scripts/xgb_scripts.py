import numpy as np
import xgboost as xgb
from sklearn.model_selection import train_test_split

# Model parameters - simple, effective defaults
MODEL_PARAMS = {
    'n_estimators': 500,
    'max_depth': 6,
    'learning_rate': 0.05,
    'subsample': 0.8,
    'colsample_bytree': 0.8,
    'reg_lambda': 1.0,
    'min_child_weight': 1,
    'objective': 'reg:squarederror',
    'n_jobs': -1,
    'random_state': 42,
    'early_stopping_rounds': 50  # Moved here for older XGBoost versions
}


def train_ensemble_model(X_train, y_train):
    # Split for early stopping validation
    X_tr, X_val, y_tr, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42
    )
    
    model = xgb.XGBRegressor(**MODEL_PARAMS)
    model.fit(
        X_tr, y_tr,
        eval_set=[(X_val, y_val)],
        verbose=False
    )
    
    return model

def predict_ensemble(model, X_test):
    predictions = model.predict(X_test)
    # Return zeros for std since we don't have an ensemble
    std_predictions = np.zeros_like(predictions)
    return predictions, std_predictions, predictions.reshape(1, -1)