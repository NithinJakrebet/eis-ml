import numpy as np
from sklearn.metrics import mean_squared_error, r2_score

def evaluate_model(y_true, y_pred):
    """ Compute RMSE and R² score for model evaluation. """
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    return {"RMSE": rmse, "R² Score": r2}
