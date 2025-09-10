from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import numpy as np
import json
import matplotlib.pyplot as plt
from datetime import datetime
import os
from plots import residual_analysis_plots, fft_analysis_plots

def evaluate_model(y_true, y_pred):
    """Basic model evaluation metrics."""
    mse = mean_squared_error(y_true, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)
    return [rmse, r2, mse, mae]

def save_results(filename, X_train, X_test, metrics, model_info, fig=None, save_plot=True, save_data=True):
    """
    Save model results including plots and metrics.
    
    Args:
        filename: Base filename for saved files
        X_train, X_test: Training and test feature matrices 
        metrics: Dictionary with rmse, r2, mse, mae
        model_info: Dictionary with model description info
        fig: matplotlib figure object to save (optional)
        save_plot: Whether to save the figure
        save_data: Whether to save metrics as JSON
    """
    # Create results directory and subdirectory for this experiment
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    experiment_dir = os.path.join(results_dir, filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    if save_plot:
        plot_path = os.path.join(experiment_dir, f'{filename}.png')
        if fig is not None:
            # Save the specific figure object
            fig.savefig(plot_path, dpi=300, bbox_inches='tight')
        else:
            # Save the current figure
            plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"Saved plot: {plot_path}")
    
    if save_data:
        results = {
            'model': model_info.get('model', 'Unknown'),
            'frequency_selection': model_info.get('frequency_selection', 'unknown'),
            'features': X_train.shape[1],
            'samples_per_feature': X_train.shape[0] / X_train.shape[1],
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
                'cycle_range': model_info.get('cycle_range', 'unknown')
            }
        }

        json_path = os.path.join(experiment_dir, f'{filename}.json')
        with open(json_path, 'w') as f:
            json.dump(results, f, indent=2)
        print(f"Saved metrics: {json_path}")

def save_residual_analysis(filename, residuals, cycle_numbers, y_pred_mean):
    """
    Save residual analysis plots and statistics.
    
    Args:
        filename: Base filename for saved files
        residuals: Residual values
        cycle_numbers: Cycle numbers for temporal analysis
        y_pred_mean: Predicted values
    """
    # Create results directory and subdirectory for this experiment
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    experiment_dir = os.path.join(results_dir, filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Create dummy y_true for the plotting function
    y_true = y_pred_mean + residuals
    
    # Create residual plots using plots module
    fig = residual_analysis_plots(residuals, cycle_numbers, y_pred_mean)
    
    # Save plot
    plot_path = os.path.join(experiment_dir, f'{filename}_residuals.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved residual plots: {plot_path}")
    
    # Show the plot
    plt.show()
    plt.close(fig)
    
    # Calculate and save statistics
    corr_cycle = np.corrcoef(cycle_numbers, residuals)[0,1]
    
    residual_stats = {
        'mean': float(np.mean(residuals)),
        'std': float(np.std(residuals)),
        'min': float(np.min(residuals)),
        'max': float(np.max(residuals)),
        'correlation_with_cycle': float(corr_cycle) if not np.isnan(corr_cycle) else 0.0,
        'temporal_bias_detected': bool(abs(corr_cycle) > 0.1) if not np.isnan(corr_cycle) else False,
        'timestamp': datetime.now().isoformat()
    }
    
    stats_path = os.path.join(experiment_dir, f'{filename}_residual_stats.json')
    with open(stats_path, 'w') as f:
        json.dump(residual_stats, f, indent=2)
    print(f"Saved residual stats: {stats_path}")
    
    return residual_stats

def save_fft_analysis(filename, residuals, cycle_numbers):
    """
    Save FFT analysis for detecting sinusoidal patterns.
    
    Args:
        filename: Base filename for saved files
        residuals: Residual values
        cycle_numbers: Cycle numbers for temporal analysis
    """
    from scipy.fft import fft, fftfreq
    
    # Create results directory and subdirectory for this experiment
    results_dir = os.path.join(os.path.dirname(__file__), '..', 'results')
    experiment_dir = os.path.join(results_dir, filename)
    os.makedirs(experiment_dir, exist_ok=True)
    
    # Sort by cycle numbers
    sorted_order = np.argsort(cycle_numbers)
    sorted_cycles = cycle_numbers[sorted_order]
    sorted_residuals = residuals[sorted_order]

    # Create FFT analysis plots using plots module and get analysis data
    fig, fft_data = fft_analysis_plots(sorted_residuals, sorted_cycles)

    # Save plot
    plot_path = os.path.join(experiment_dir, f'{filename}_fft.png')
    fig.savefig(plot_path, dpi=300, bbox_inches='tight')
    print(f"Saved FFT analysis: {plot_path}")
    
    # Show the plot
    plt.show()
    plt.close(fig)

    # Add timestamp to the analysis data
    fft_data['timestamp'] = datetime.now().isoformat()

    json_path = os.path.join(experiment_dir, f'{filename}_fft_analysis.json')
    with open(json_path, 'w') as f:
        json.dump(fft_data, f, indent=2)
    print(f"Saved FFT data: {json_path}")
    
    return fft_data