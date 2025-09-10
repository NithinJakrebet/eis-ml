from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import config
import data_pipeline
import model_training

def evaluate_model(y_true, y_pred):
    """Basic model evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return rmse, r2, mse, mae

def test_on_later_cycles(trained_models, test_folder, start_cycle=50, max_cycles=48):
    """
    Test model on later cycles from longer datasets for temporal validation.
    
    Args:
        trained_models: List of trained models from ensemble
        test_folder (str): Folder containing test dataset
        start_cycle (int): Starting cycle number for test
        max_cycles (int): Maximum number of cycles to test
        
    Returns:
        dict: Test results with metrics
    """
    print(f"Testing model on cycles {start_cycle}-{start_cycle+max_cycles-1} from {test_folder}")
    
    try:
        # Temporarily change config to load from test folder
        original_folder = config.DEFAULT_DATA_FOLDER
        config.DEFAULT_DATA_FOLDER = test_folder
        
        # Load all data from test folder
        X_test_full, _, y_test_full, _ = data_pipeline.load_and_prepare_data()
        
        # Restore original config
        config.DEFAULT_DATA_FOLDER = original_folder
        
        # Calculate available cycles
        total_cycles = len(X_test_full)
        end_cycle = min(start_cycle + max_cycles - 1, total_cycles)
        
        if start_cycle > total_cycles:
            print(f"Error: Start cycle {start_cycle} > available cycles {total_cycles}")
            return None
        
        # Extract later cycles
        start_idx = start_cycle - 1  # Convert to 0-based indexing
        end_idx = end_cycle
        
        X_later = X_test_full[start_idx:end_idx]
        y_later = y_test_full[start_idx:end_idx]
        
        print(f"Loaded cycles {start_cycle}-{end_cycle} ({len(X_later)} cycles)")
        print(f"Capacity range: {y_later.min():.1f} - {y_later.max():.1f} mA.h")
        
        # Make predictions
        y_pred_mean, y_pred_std, _ = model_training.predict_ensemble(trained_models, X_later)
        
        # Evaluate performance
        rmse, r2, mse, mae = evaluate_model(y_later, y_pred_mean)
        
        print(f"\nTemporal Validation Performance ({test_folder}, cycles {start_cycle}-{end_cycle}):")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R²: {r2:.4f}")
        print(f"  MSE: {mse:.4f}")
        print(f"  MAE: {mae:.4f}")
        
        return {
            'dataset': test_folder,
            'cycle_range': f"{start_cycle}-{end_cycle}",
            'cycles_tested': len(X_later),
            'rmse': rmse,
            'r2': r2,
            'mse': mse,
            'mae': mae,
            'capacity_range': (y_later.min(), y_later.max())
        }
        
    except Exception as e:
        print(f"Error testing later cycles in {test_folder}: {e}")
        return None

def analyze_prediction_failure(y_true, y_pred, title="Prediction Analysis"):
    """
    Analyze prediction errors to understand model limitations.
    
    Args:
        y_true: True values
        y_pred: Predicted values
        title (str): Title for the analysis
        
    Returns:
        dict: Analysis results
    """
    # Calculate prediction characteristics
    bias = np.mean(y_pred - y_true)
    mae = np.mean(np.abs(y_pred - y_true))
    rmse = np.sqrt(np.mean((y_pred - y_true)**2))
    
    print(f"\n=== {title} ===")
    print(f"Bias (systematic error): {bias:.2f} mA.h")
    print(f"MAE (average error): {mae:.2f} mA.h") 
    print(f"RMSE (error magnitude): {rmse:.2f} mA.h")
    
    # Check if model is consistently over/under predicting
    if abs(bias) > mae * 0.5:
        if bias > 0:
            interpretation = "Model OVER-predicts capacity (thinks batteries are healthier)"
        else:
            interpretation = "Model UNDER-predicts capacity (thinks batteries are more degraded)"
    else:
        interpretation = "Model shows balanced errors (no systematic bias)"
    
    print(f"Interpretation: {interpretation}")
    
    # Analyze error patterns
    errors = y_pred - y_true
    if np.std(errors) > mae:
        error_pattern = "High error variability - model uncertainty increases"
    else:
        error_pattern = "Consistent error pattern - systematic model limitation"
    
    print(f"Error pattern: {error_pattern}")
    
    # Capacity range analysis
    pred_range = y_pred.max() - y_pred.min()
    true_range = y_true.max() - y_true.min()
    
    print(f"Predicted capacity range: {pred_range:.1f} mA.h")
    print(f"Actual capacity range: {true_range:.1f} mA.h")
    
    return {
        'bias': bias,
        'mae': mae,
        'rmse': rmse,
        'interpretation': interpretation,
        'error_pattern': error_pattern,
        'pred_range': pred_range,
        'true_range': true_range
    }
