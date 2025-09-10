import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def degradation(df: pd.DataFrame):
      capacity = df['Capacity/mA.h']
      cycle_number = df['cycle number']
      
      plt.figure(figsize=(8, 8))
      plt.scatter(cycle_number, capacity, alpha=0.6)
      plt.xlabel('Cycle Number')
      plt.ylabel('Capacity (mA.h)')
      plt.title('Capacity vs. Cycle Number')
      plt.show()

      
      
def unique_frequencies(unique_freqs: np.array):
      plt.figure(figsize=(12, 5))

      # Linear scale plot
      plt.subplot(1, 2, 1)
      plt.plot(unique_freqs, 'o-', label='Frequency')
      plt.xlabel('Index')
      plt.ylabel('Frequency (Hz)')
      plt.title('Unique Frequencies (Linear Scale)')
      plt.legend()

      # Logarithmic scale plot (y-axis on log scale)
      plt.subplot(1, 2, 2)
      plt.semilogy(unique_freqs, 'o-', label='Frequency')
      plt.xlabel('Index')
      plt.ylabel('Frequency (Hz)')
      plt.title('Unique Frequencies (Logarithmic Scale)')
      plt.legend()

      plt.tight_layout()
      plt.show()

      
def nyquist(df: pd.DataFrame):
    # Prepare full dataset values
    Re_Z_full = df['Re(Z)/Ohm'].values
    Im_Z_full = df['Im(Z)/Ohm'].values

    # Prepare filtered dataset values
    filtered_df = df.loc[(df['Ns'].isin([1, 6])) & (df['cycle number'] != 0)].copy()
    Re_Z_filtered = filtered_df['Re(Z)/Ohm'].values
    Im_Z_filtered = filtered_df['Im(Z)/Ohm'].values

    # Create subplots: two columns side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot full dataset Nyquist plot on the first subplot
    axes[0].plot(Re_Z_full, Im_Z_full, 'o', markersize=5, alpha=0.7, label="Impedance Data")
    axes[0].set_xlabel('Re(Z) / Ohm')
    axes[0].set_ylabel('Im(Z) / Ohm')
    axes[0].set_title('Nyquist Plot of Battery Impedance')
    axes[0].grid(True)
    axes[0].axis('equal')
    axes[0].legend()

    # Plot filtered dataset Nyquist plot on the second subplot
    axes[1].plot(Re_Z_filtered, Im_Z_filtered, 'o', markersize=5, alpha=0.7, label="Impedance Data")
    axes[1].set_xlabel('Re(Z) / Ohm')
    axes[1].set_ylabel('Im(Z) / Ohm')
    axes[1].set_title('Nyquist Plot of Battery Impedance (EIS States)')
    axes[1].grid(True)
    axes[1].axis('equal')
    axes[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


def model_predictions(y_true, y_pred_mean, y_pred_std=None, title_prefix="Model"):
    fig = plt.figure(figsize=(15, 5))
    
    # Scatter plot: Predicted vs. True
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred_mean, alpha=0.7, edgecolors='k', label='Predictions')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Discharge Capacity (mA.h)')
    plt.ylabel('Predicted Discharge Capacity (mA.h)')
    plt.title(f'{title_prefix}: Predicted vs. True')
    plt.legend()
    plt.grid(True)
    
    # Residual plot: Error vs. True values
    residuals = y_true - y_pred_mean
    plt.subplot(1, 3, 2)
    plt.scatter(y_true, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('True Discharge Capacity (mA.h)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'{title_prefix}: Residual Plot')
    plt.grid(True)
    
    # Print summary statistics
    print(f"\n{title_prefix} Prediction Analysis:")
    print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
    print(f"  Root Mean Square Error: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"  Residual Std: {np.std(residuals):.4f}")
    if y_pred_std is not None:
        print(f"  Mean Prediction Uncertainty: {np.mean(y_pred_std):.4f}")
    
    plt.tight_layout()
    return fig


def ensemble_uncertainty(y_true, y_pred_mean, y_pred_std, title="Ensemble Predictions with Uncertainty"):
    """
    Plot predictions with uncertainty bands from ensemble models.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred_mean (np.ndarray): Mean predictions
        y_pred_std (np.ndarray): Standard deviation of predictions
        title (str): Plot title
    """
    # Sort for better visualization
    sort_idx = np.argsort(y_true)
    y_true_sorted = y_true[sort_idx]
    y_pred_mean_sorted = y_pred_mean[sort_idx]
    y_pred_std_sorted = y_pred_std[sort_idx]
    
    plt.figure(figsize=(12, 6))
    
    # Plot with uncertainty bands
    plt.subplot(1, 2, 1)
    plt.scatter(y_true_sorted, y_pred_mean_sorted, alpha=0.6, label='Predictions')
    plt.fill_between(y_true_sorted, 
                     y_pred_mean_sorted - 2*y_pred_std_sorted,
                     y_pred_mean_sorted + 2*y_pred_std_sorted,
                     alpha=0.3, label='95% Confidence')
    plt.plot([y_true.min(), y_true.max()], [y_true.min(), y_true.max()], 'r--', lw=2, label='Perfect Prediction')
    plt.xlabel('True Discharge Capacity (mA.h)')
    plt.ylabel('Predicted Discharge Capacity (mA.h)')
    plt.title(f'{title}: Predictions with Uncertainty')
    plt.legend()
    plt.grid(True)
    
    # Uncertainty vs prediction error
    prediction_error = np.abs(y_true - y_pred_mean)
    plt.subplot(1, 2, 2)
    plt.scatter(y_pred_std, prediction_error, alpha=0.6)
    plt.xlabel('Prediction Uncertainty (Std)')
    plt.ylabel('Absolute Prediction Error')
    plt.title('Uncertainty vs. Prediction Error')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()


def residual_analysis_plots(residuals, cycle_numbers, y_pred_mean):
    """
    Create residual analysis plots.
    
    Args:
        residuals: Residual values
        cycle_numbers: Cycle numbers for temporal analysis
        y_pred_mean: Predicted values
    
    Returns:
        matplotlib figure object
    """
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))

    # Plot 1: Residuals vs Cycle Number
    axes[0].scatter(cycle_numbers, residuals, alpha=0.6, s=20)
    axes[0].set_xlabel('Cycle Number')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Cycle Number')
    axes[0].grid(True, alpha=0.3)

    # Plot 2: Residuals vs Predicted Values
    axes[1].scatter(y_pred_mean, residuals, alpha=0.6, s=20)
    axes[1].set_xlabel('Predicted Capacity (mA.h)')
    axes[1].set_ylabel('Residuals')
    axes[1].set_title('Residuals vs Predictions')
    axes[1].axhline(y=0, color='r', linestyle='--', alpha=0.7)
    axes[1].grid(True, alpha=0.3)

    # Plot 3: Sequential residuals to see patterns
    axes[2].plot(range(len(residuals)), residuals, 'b-', linewidth=1)
    axes[2].set_xlabel('Sample Index')
    axes[2].set_ylabel('Residuals')
    axes[2].set_title('Sequential Residual Pattern')
    axes[2].grid(True, alpha=0.3)

    plt.tight_layout()
    return fig


def fft_analysis_plots(residuals, cycle_numbers):
    """
    Create FFT analysis plots for detecting sinusoidal patterns.
    
    Args:
        residuals: Residual values
        cycle_numbers: Cycle numbers for temporal analysis
    
    Returns:
        tuple: (matplotlib figure object, fft analysis data dict)
    """
    from scipy.fft import fft, fftfreq
    
    # Sort by cycle numbers
    sorted_order = np.argsort(cycle_numbers)
    sorted_cycles = cycle_numbers[sorted_order]
    sorted_residuals = residuals[sorted_order]

    # FFT analysis
    fft_result = fft(sorted_residuals)
    frequencies = fftfreq(len(sorted_residuals), d=1)  # 1 cycle spacing
    magnitude = np.abs(fft_result)

    # Find dominant frequency (skip DC component)
    pos_freqs = frequencies[1:len(frequencies)//2]
    pos_magnitudes = magnitude[1:len(magnitude)//2]
    dominant_idx = np.argmax(pos_magnitudes)
    dominant_freq = pos_freqs[dominant_idx]
    dominant_period = 1 / dominant_freq if dominant_freq > 0 else float('inf')

    # Create plots
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(sorted_cycles, sorted_residuals, 'b-', linewidth=1)
    axes[0].set_xlabel('Cycle Number')
    axes[0].set_ylabel('Residuals')
    axes[0].set_title('Residuals vs Cycle (Check for Waves)')
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(pos_freqs[:50], pos_magnitudes[:50])  # Show first 50 frequencies
    axes[1].set_xlabel('Frequency (cycles⁻¹)')
    axes[1].set_ylabel('Magnitude')
    axes[1].set_title('Frequency Spectrum')
    axes[1].grid(True, alpha=0.3)
    axes[1].axvline(x=dominant_freq, color='r', linestyle='--', alpha=0.7, label=f'Peak at {dominant_freq:.3f}')
    axes[1].legend()

    plt.tight_layout()
    
    # Return both figure and analysis data
    fft_data = {
        'dominant_frequency': float(dominant_freq) if not np.isnan(dominant_freq) else 0.0,
        'dominant_period': float(dominant_period) if not np.isinf(dominant_period) and not np.isnan(dominant_period) else 0.0,
        'has_periodic_pattern': bool(5 < dominant_period < 50) if not np.isinf(dominant_period) and not np.isnan(dominant_period) else False,
    }
    
    return fig, fft_data

