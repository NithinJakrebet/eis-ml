import numpy as np
import pandas as pd
import xgboost as xgb
import config
import os
import glob

def compute_sample_weights(y_train, n_bins=10, method='inverse_frequency'):
    """
    Compute sample weights to handle class imbalance.
    
    Args:
        y_train: Training targets
        n_bins: Number of bins for binning continuous targets
        method: Weighting method ('inverse_frequency' or 'balanced')
    
    Returns:
        sample_weights: Array of weights for each training sample
    """
    # Create capacity bins
    y_bins = pd.qcut(y_train, q=n_bins, duplicates='drop')
    
    if method == 'inverse_frequency':
        # Weight inversely proportional to bin frequency
        bin_counts = y_bins.value_counts()
        bin_weights = 1.0 / bin_counts
        # Normalize weights to have mean = 1
        bin_weights = bin_weights / bin_weights.mean()
        sample_weights = y_bins.map(bin_weights, na_action=None).values
        
    elif method == 'balanced':
        # sklearn-style balanced weighting
        bin_counts = y_bins.value_counts()
        n_samples = len(y_train)
        n_classes = len(bin_counts)
        
        bin_weights = n_samples / (n_classes * bin_counts)
        sample_weights = y_bins.map(bin_weights, na_action=None).values
    
    else:
        raise ValueError(f"Unknown weighting method: {method}")
    
    print(f"Sample weights computed using {method}:")
    print(f"  Weight range: {sample_weights.min():.3f} - {sample_weights.max():.3f}")
    print(f"  Mean weight: {sample_weights.mean():.3f}")
    
    return sample_weights

def _get_improved_model_params():
    """Get improved model parameters to reduce mean-hugging behavior."""
    return {
        'n_estimators': 2000,
        'max_depth': 8,
        'learning_rate': 0.03,
        'min_child_weight': 1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'reg_alpha': 0,
        'reg_lambda': 1,
        'gamma': 0,
        'objective': 'reg:squarederror',
        'random_state': 42
    }

def _prepare_validation_split(X_train, y_train, sample_weights, validation_split=0.15):
    """Prepare validation split for early stopping."""
    if validation_split <= 0:
        return X_train, y_train, sample_weights, None, None
    
    val_size = int(len(X_train) * validation_split)
    val_indices = np.random.choice(len(X_train), val_size, replace=False)
    train_indices = np.setdiff1d(np.arange(len(X_train)), val_indices)
    
    X_val, y_val = X_train[val_indices], y_train[val_indices]
    X_train_fit, y_train_fit = X_train[train_indices], y_train[train_indices]
    
    weights_fit = sample_weights[train_indices] if sample_weights is not None else None
    
    return X_train_fit, y_train_fit, weights_fit, X_val, y_val

def train_ensemble_model(
    X_train, 
    y_train, 
    n_models=None, 
    model_params=None, 
    use_improvements=False, 
    use_sample_weights=False, 
    weight_method='inverse_frequency'
):
    """Enhanced ensemble training with optional improvements."""
    if n_models is None: n_models = config.NUM_ENSEMBLE
    
    # Use improved parameters if requested
    if use_improvements and model_params is None:
        model_params = _get_improved_model_params()
    elif model_params is None:
        model_params = config.MODEL_PARAMS.copy()
    
    # Compute sample weights if requested
    sample_weights = None
    if use_sample_weights:
        sample_weights = compute_sample_weights(y_train, method=weight_method)
    
    # Prepare validation split if using improvements
    if use_improvements:
        X_train_fit, y_train_fit, weights_fit, X_val, y_val = _prepare_validation_split(
            X_train, y_train, sample_weights, validation_split=0.15
        )
    else:
        X_train_fit, y_train_fit = X_train, y_train
        weights_fit = sample_weights
        X_val, y_val = None, None
    
    ensemble_models = []
        
    for i in range(n_models):
        model_params_copy = model_params.copy()
        model_params_copy['random_state'] = model_params_copy.get('random_state', 42) + i
        
        # Add early stopping to model parameters if using improvements
        if use_improvements and X_val is not None:
            model_params_copy['early_stopping_rounds'] = 100
            model_params_copy['eval_metric'] = 'rmse'
        
        model = xgb.XGBRegressor(**model_params_copy)
        
        # Prepare fit parameters
        fit_params = {}
        if weights_fit is not None:
            fit_params['sample_weight'] = weights_fit
        
        # Add validation set if available
        if use_improvements and X_val is not None:
            fit_params['eval_set'] = [(X_val, y_val)]
            fit_params['verbose'] = False
        
        model.fit(X_train_fit, y_train_fit, **fit_params)
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

def analyze_covariate_shift(X_train, X_test, feature_names=None):
    """
    Analyze potential covariate shift between train and test sets.
    """
    n_features = X_train.shape[1]
    if feature_names is None:
        feature_names = [f"feature_{i}" for i in range(n_features)]
    
    shift_analysis = []
    
    for i in range(n_features):
        train_min, train_max = X_train[:, i].min(), X_train[:, i].max()
        test_min, test_max = X_test[:, i].min(), X_test[:, i].max()
        
        # Fraction of test samples outside train range
        outside_range = np.sum((X_test[:, i] < train_min) | (X_test[:, i] > train_max))
        outside_fraction = outside_range / len(X_test)
        
        shift_analysis.append({
            'feature': feature_names[i],
            'train_range': (train_min, train_max),
            'test_range': (test_min, test_max),
            'outside_train_range': outside_fraction
        })
    
    shift_df = pd.DataFrame(shift_analysis)
    
    # Summary statistics
    high_shift_features = shift_df[shift_df['outside_train_range'] > 0.1]
    
    print("=== Covariate Shift Analysis ===")
    print(f"Features with >10% test samples outside train range: {len(high_shift_features)}")
    
    if len(high_shift_features) > 0:
        print("High shift features:")
        for _, row in high_shift_features.iterrows():
            print(f"  {row['feature']}: {row['outside_train_range']:.1%} outside range")
    
    return shift_df

def analyze_residuals_by_capacity(y_true, y_pred, n_bins=10):
    """
    Analyze residuals across different capacity ranges.
    """
    residuals = y_true - y_pred
    
    # Create capacity bins
    capacity_bins = pd.qcut(y_true, q=n_bins, duplicates='drop')
    
    residual_analysis = []
    for bin_label in capacity_bins.categories:
        mask = (capacity_bins == bin_label)
        bin_residuals = residuals[mask]
        
        residual_analysis.append({
            'capacity_range': str(bin_label),
            'n_samples': len(bin_residuals),
            'mean_residual': bin_residuals.mean(),
            'std_residual': bin_residuals.std(),
            'mean_abs_residual': np.abs(bin_residuals).mean()
        })
    
    residual_df = pd.DataFrame(residual_analysis)
    
    print("=== Residual Analysis by Capacity ===")
    print(residual_df.round(3))
    
    return residual_df
