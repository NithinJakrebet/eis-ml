from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

def evaluate_model(y_true, y_pred) -> dict:
    """
    Computes MSE, RMSE, R² score, and MAE for model evaluation.
    Prints them and returns a dictionary of the metrics.
    """
    mse = mean_squared_error(y_true, y_pred)
    rmse = mean_squared_error(y_true, y_pred, squared=False)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    print(f"RMSE: {rmse:.4f}, R² Score: {r2:.4f}, MSE: {mse:.4f}, MAE: {mae:.4f}")

    return {
        "RMSE": rmse,
        "R2": r2,
        "MSE": mse,
        "MAE": mae
    }
