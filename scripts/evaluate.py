from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
from plots import residual_analysis_plots
import config

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return [rmse, r2, mse, mae]

def save_results(filename, X_train, X_test, metrics, model_info, fig=None):
    experiment_dir = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    plot_path = os.path.join(experiment_dir, f'{filename}.png')
    if fig is not None:
        fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    else:
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    results = {
        'model': model_info.get('model', 'Unknown'),
        'frequency_selection': model_info.get('frequency_selection', 'unknown'),
        'features': X_train.shape[1],
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae'])
        },
        'data_info': {
            'train_samples': X_train.shape[0],
            'test_samples': X_test.shape[0],
        }
    }

    json_path = os.path.join(experiment_dir, f'{filename}.json')
    with open(json_path, 'w') as f:
        json.dump(results, f, indent=2)

def save_residual_analysis(filename, residuals, cycle_numbers, y_pred_mean):
    experiment_dir = os.path.join(config.RESULTS_DIR, filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    fig = residual_analysis_plots(residuals, cycle_numbers, y_pred_mean)
    
    plot_path = os.path.join(experiment_dir, f'{filename}_residuals.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    plt.show()
    plt.close(fig)
    
    corr_cycle = np.corrcoef(cycle_numbers, residuals)[0,1]
    
    residual_stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'correlation_with_cycle': float(corr_cycle) if not np.isnan(corr_cycle) else 0.0,
        'temporal_bias_detected': bool(abs(corr_cycle) > 0.1) if not np.isnan(corr_cycle) else False,
    }
    
    stats_path = os.path.join(experiment_dir, f'{filename}_residual_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(residual_stats, f, indent=2)
    
    return residual_stats