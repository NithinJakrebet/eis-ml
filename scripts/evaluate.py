from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os

def evaluate_model(y_true, y_pred):
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return [rmse, r2, mse, mae]

def save_results(filename, metrics, model_info, fig=None):
    experiment_dir = os.path.join("../results", filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    plot_path = os.path.join(experiment_dir, f'{filename}.png')
    if fig is not None: fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    else: plt.savefig(plot_path, dpi=300, bbox_inches='tight')
    
    results = {
        'model': model_info.get('model', 'Unknown'),
        'frequency_selection': model_info.get('frequency_selection', 'unknown'),
        'timestamp': datetime.now().isoformat(),
        'metrics': {
            'rmse': float(metrics['rmse']),
            'r2': float(metrics['r2']),
            'mse': float(metrics['mse']),
            'mae': float(metrics['mae'])
        }
    }

    json_path = os.path.join(experiment_dir, f'{filename}.json')
    with open(json_path, 'w') as f: json.dump(results, f, indent=2)