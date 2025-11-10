import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def degradation(channel, df: pd.DataFrame):
    ns8 = df[df['Ns'] == 8]
    if ns8.empty:
        raise ValueError(f"No rows with Ns == 8 found for channel {channel}")

    cap8 = ns8.groupby('cycle number')['Capacity/mA.h'].last().sort_index()

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.plot(cap8.index, cap8.values, 'o', markersize=5, alpha=0.9, label='Ns = 8')
    ax.set_xlabel('Cycle Number')
    ax.set_ylabel('Capacity (mA.h)')
    ax.set_title(f'{channel}: Ns = 8 Capacity vs Cycle Number')
    ax.grid(True, alpha=0.3)
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig
    
    
    
def ard_summary(freqs_hz_ns_1, freqs_hz_ns_6, re_mean, re_std, im_mean, im_std, save_path=None):
    fig, ax = plt.subplots(figsize=(16, 8))
    
    ns1_idx = list(range(len(freqs_hz_ns_1)))
    ns6_idx = list(range(len(freqs_hz_ns_1), len(freqs_hz_ns_1) + len(freqs_hz_ns_6)))
    
    ax.semilogx(freqs_hz_ns_1, re_mean[ns1_idx], 'o-', color='blue', 
                label='Re(Z) Ns1 (discharged)', markersize=5)
    ax.fill_between(freqs_hz_ns_1, 
                     (re_mean - re_std)[ns1_idx], 
                     (re_mean + re_std)[ns1_idx], 
                     alpha=0.2, color='blue')
    
    ax.semilogx(freqs_hz_ns_1, im_mean[ns1_idx], 's--', color='cyan', 
                label='-Im(Z) Ns1 (discharged)', markersize=5)
    ax.fill_between(freqs_hz_ns_1, 
                     (im_mean - im_std)[ns1_idx], 
                     (im_mean + im_std)[ns1_idx], 
                     alpha=0.2, color='cyan')
    
    ax.semilogx(freqs_hz_ns_6, re_mean[ns6_idx], '^-', color='red', 
                label='Re(Z) Ns6 (charged)', markersize=5)
    ax.fill_between(freqs_hz_ns_6, 
                     (re_mean - re_std)[ns6_idx], 
                     (re_mean + re_std)[ns6_idx], 
                     alpha=0.2, color='red')
    
    ax.semilogx(freqs_hz_ns_6, im_mean[ns6_idx], 'v--', color='orange', 
                label='-Im(Z) Ns6 (charged)', markersize=5)
    ax.fill_between(freqs_hz_ns_6, 
                     (im_mean - im_std)[ns6_idx], 
                     (im_mean + im_std)[ns6_idx], 
                     alpha=0.2, color='orange')
    
    # Add frequency labels for each point
    # Label Ns1 frequencies
    for i, freq in enumerate(freqs_hz_ns_1):
        # Get the max height at this frequency (from all 4 curves)
        y_max = max(
            re_mean[ns1_idx[i]] + re_std[ns1_idx[i]],
            im_mean[ns1_idx[i]] + im_std[ns1_idx[i]]
        )
        # Format frequency label
        if freq < 1:
            label = f'{freq:.2f}'
        elif freq < 100:
            label = f'{freq:.1f}'
        else:
            label = f'{int(freq)}'
        
        ax.text(freq, y_max, label, fontsize=7, ha='center', va='bottom', 
                rotation=45, color='darkblue')
    
    # Label Ns6 frequencies
    for i, freq in enumerate(freqs_hz_ns_6):
        # Get the max height at this frequency
        y_max = max(
            re_mean[ns6_idx[i]] + re_std[ns6_idx[i]],
            im_mean[ns6_idx[i]] + im_std[ns6_idx[i]]
        )
        # Format frequency label
        if freq < 1:
            label = f'{freq:.2f}'
        elif freq < 100:
            label = f'{freq:.1f}'
        else:
            label = f'{int(freq)}'
        
        ax.text(freq, y_max, label, fontsize=7, ha='center', va='bottom', 
                rotation=45, color='darkred')
    
    ax.set_xlabel("Frequency (Hz)", fontsize=12)
    ax.set_ylabel("Relative importance (exp(-ℓ))", fontsize=12)
    ax.set_title("ARD Frequency Importance: Dual Ns State Vector (Ns1=Discharged, Ns6=Charged)\n14-Fold LOSO Cross-Validation", 
                 fontsize=13)
    ax.grid(True, which='both', ls='--', alpha=0.4)
    ax.legend(loc='best', fontsize=9)
    plt.tight_layout()
    
    if save_path:
        fig.savefig(save_path, dpi=180, bbox_inches='tight')
    
    return fig
    
    
    
    
    
    

      
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

      
def nyquist(df: pd.DataFrame, title_prefix=""):
    Re_Z_full = np.asarray(df["Re(Z)/Ohm"].astype(float).values)
    Im_Z_full = np.asarray(df["-Im(Z)/Ohm"].astype(float).values)

    fig, ax = plt.subplots(figsize=(6, 6))

    ax.scatter(Re_Z_full, Im_Z_full, s=30, alpha=0.75)
    ax.set_xlabel('Re(Z) / Ohm')
    ax.set_ylabel('-Im(Z) / Ohm')
    ax.set_title(f'{title_prefix} Nyquist Plot of Battery Impedance' if title_prefix else 'Nyquist Plot of Battery Impedance')
    ax.grid(True, linestyle='--', alpha=0.5)
    ax.axis('equal')
    ax.legend()
    plt.tight_layout()
    plt.show()

    return fig, ax


def model_predictions(y_true, y_pred_mean, y_pred_std=None, title_prefix="Model"):
    """
    Plots model predictions:
      - Predicted vs. True scatter (with optional 95% CI band)
      - Residuals vs. True
      - Displays mean predictive uncertainty if provided
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2 = r2_score(y_true, y_pred_mean)
    mse = mean_squared_error(y_true, y_pred_mean)
    mae = mean_absolute_error(y_true, y_pred_mean)
    
    fig = plt.figure(figsize=(15, 5))
    
    # --- 1. Predicted vs. True ---
    plt.subplot(1, 3, 1)
    plt.scatter(y_true, y_pred_mean, alpha=0.7, edgecolors='k', label='Predictions')

    # Perfect prediction line
    min_val = min(y_true.min(), y_pred_mean.min())
    max_val = max(y_true.max(), y_pred_mean.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--', lw=2, label='Perfect Prediction')

    # --- Optional CI band (95%) ---
    if y_pred_std is not None:
        ci_low = y_pred_mean - 1.96 * y_pred_std
        ci_high = y_pred_mean + 1.96 * y_pred_std

        # Sort for proper band drawing
        sorted_idx = np.argsort(y_true)
        plt.fill_between(
            y_true[sorted_idx],
            ci_low[sorted_idx],
            ci_high[sorted_idx],
            color='orange',
            alpha=0.3,
            label='95% CI'
        )

    plt.xlabel('True Discharge Capacity (mA.h)')
    plt.ylabel('Predicted Discharge Capacity (mA.h)')
    plt.title(f'{title_prefix}: Predicted vs. True')
    plt.legend()
    plt.grid(True)
    
    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}'
    plt.text(0.05, 0.95, metrics_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='top',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # --- 2. Residual plot ---
    residuals = y_true - y_pred_mean
    plt.subplot(1, 3, 2)
    plt.scatter(y_true, residuals, alpha=0.7, edgecolors='k')
    plt.axhline(0, color='r', linestyle='--', lw=2)
    plt.xlabel('True Discharge Capacity (mA.h)')
    plt.ylabel('Residuals (True - Predicted)')
    plt.title(f'{title_prefix}: Residual Plot')
    plt.grid(True)
    
    # --- 3. Uncertainty summary ---
    if y_pred_std is not None:
        mean_unc = np.mean(y_pred_std)
        plt.suptitle(f"{title_prefix} (Mean Prediction Uncertainty = {mean_unc:.4f})", fontsize=14)
        print(f"  Mean Prediction Uncertainty: {mean_unc:.4f}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])  # leave space for suptitle
    return fig



def residual_analysis_plots(residuals, cycle_numbers, y_pred_mean, y_true=None):
    """
    Plot residual analysis with optional metrics display.
    
    Args:
        residuals: Residual values (y_true - y_pred)
        cycle_numbers: Cycle numbers for x-axis
        y_pred_mean: Predicted values
        y_true: Optional true values to calculate metrics
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
    
    # Add metrics if y_true is provided
    if y_true is not None:
        from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
        rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
        r2 = r2_score(y_true, y_pred_mean)
        mse = mean_squared_error(y_true, y_pred_mean)
        mae = mean_absolute_error(y_true, y_pred_mean)
        
        metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}'
        axes[0].text(0.05, 0.95, metrics_text, transform=axes[0].transAxes, 
                     fontsize=9, verticalalignment='top',
                     bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))

    plt.tight_layout()
    return fig


def capacity_vs_cycle(y_true, y_pred_mean, cycle_numbers, y_pred_std=None, title_prefix="Model"):
    """
    Plot actual vs predicted capacity over cycle number with optional confidence interval.
    Uses line plots with markers for better visualization of degradation trends.
    
    Args:
        y_true: True capacity values
        y_pred_mean: Predicted capacity values
        cycle_numbers: Cycle numbers for x-axis
        y_pred_std: Optional standard deviation for 95% confidence interval
        title_prefix: Prefix for plot title
        
    Returns:
        fig: Matplotlib figure object
    """
    from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    
    # Calculate metrics
    rmse = np.sqrt(mean_squared_error(y_true, y_pred_mean))
    r2 = r2_score(y_true, y_pred_mean)
    mse = mean_squared_error(y_true, y_pred_mean)
    mae = mean_absolute_error(y_true, y_pred_mean)
    
    fig = plt.figure(figsize=(10, 6))
    
    # Sort by cycle number for proper line drawing
    sorted_idx = np.argsort(cycle_numbers)
    sorted_cycles = cycle_numbers[sorted_idx]
    sorted_true = y_true[sorted_idx]
    sorted_pred = y_pred_mean[sorted_idx]
    
    # Plot actual and predicted with line connections
    plt.plot(sorted_cycles, sorted_true, 'o-', label='Actual', markersize=5)
    plt.plot(sorted_cycles, sorted_pred, 's--', label='Predicted', markersize=5)
    
    # Add 95% confidence interval if standard deviation is provided
    if y_pred_std is not None:
        sorted_std = y_pred_std[sorted_idx]
        plt.fill_between(
            sorted_cycles,
            sorted_pred - 1.96 * sorted_std,
            sorted_pred + 1.96 * sorted_std,
            alpha=0.2,
            label='95% CI'
        )
    
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (mA.h)')
    plt.title(f'{title_prefix}: Capacity vs Cycle Number')
    plt.legend(loc='best')
    plt.grid(True, alpha=0.3)
    
    # Add metrics text box
    metrics_text = f'R² = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\nMSE = {mse:.4f}'
    plt.text(0.05, 0.05, metrics_text, transform=plt.gca().transAxes, 
             fontsize=10, verticalalignment='bottom',
             bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    plt.tight_layout()
    
    return fig



def multicell_loso_predictions(cell_metrics, title="LOSO Cross-Validation", figsize=(12, 10)):
    fig, ax = plt.subplots(figsize=figsize)
    colors = plt.cm.tab10(np.linspace(0, 1, len(cell_metrics)))
    
    min_val, max_val = float('inf'), float('-inf')
    
    for i, (cell_name, y_true, y_pred, r2, rmse, mae) in enumerate(cell_metrics):
        min_val = min(min_val, y_true.min(), y_pred.min())
        max_val = max(max_val, y_true.max(), y_pred.max())
        
        ax.scatter(y_true, y_pred, alpha=0.7, s=40, color=colors[i], 
                   label=f'{cell_name} (R²={r2:.3f})', edgecolors='white', linewidth=0.5)
    
    ax.plot([min_val, max_val], [min_val, max_val], 'k--', lw=3, alpha=0.8, label='Perfect Prediction')
    
    ax.set_xlabel('Actual Capacity', fontsize=14)
    ax.set_ylabel('Predicted Capacity', fontsize=14)
    ax.set_title(title, fontsize=16)
    ax.grid(True, alpha=0.3)
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
    ax.set_aspect('equal', adjustable='box')
    
    avg_r2 = np.mean([r2 for _, _, _, r2, _, _ in cell_metrics])
    avg_rmse = np.mean([rmse for _, _, _, _, rmse, _ in cell_metrics])
    stats_text = f'Average R² = {avg_r2:.3f}\nAverage RMSE = {avg_rmse:.4f}'
    ax.text(0.02, 0.98, stats_text, transform=ax.transAxes, fontsize=12,
            verticalalignment='top', bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    
    plt.tight_layout()
    return fig, ax

