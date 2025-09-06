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
    """
    Plot model predictions vs true values with residual analysis.
    
    Args:
        y_true (np.ndarray): True target values
        y_pred_mean (np.ndarray): Mean predictions
        y_pred_std (np.ndarray, optional): Standard deviation of predictions (for uncertainty)
        title_prefix (str): Prefix for plot titles
    """
    plt.figure(figsize=(15, 5))
    
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
    
    # Histogram of residuals
    plt.subplot(1, 3, 3)
    plt.hist(residuals, bins=20, alpha=0.7, edgecolor='black')
    plt.axvline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('Residuals (True - Predicted)')
    plt.ylabel('Frequency')
    plt.title(f'{title_prefix}: Residual Distribution')
    plt.grid(True)
    
    plt.tight_layout()
    plt.show()
    
    # Print summary statistics
    print(f"\n{title_prefix} Prediction Analysis:")
    print(f"  Mean Absolute Error: {np.mean(np.abs(residuals)):.4f}")
    print(f"  Root Mean Square Error: {np.sqrt(np.mean(residuals**2)):.4f}")
    print(f"  Residual Std: {np.std(residuals):.4f}")
    if y_pred_std is not None:
        print(f"  Mean Prediction Uncertainty: {np.mean(y_pred_std):.4f}")


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

