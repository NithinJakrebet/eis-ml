import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
import pandas as pd
import numpy as np

def degradation(channel, df: pd.DataFrame):
    ns6 = df[df['Ns'] == 6]
    cap6 = ns6.groupby('cycle number')['Capacity/mA.h'].mean()

    ns8 = df[df['Ns'] == 8]
    cap8 = ns8.groupby('cycle number')['Capacity/mA.h'].last()

    common_cycles = cap6.index.intersection(cap8.index)
    c6 = cap6.loc[common_cycles]
    c8 = cap8.loc[common_cycles]

    pearson_r, _ = pearsonr(c6, c8)
    r2 = r2_score(c6, c8)

    plt.figure(figsize=(8, 8))
    plt.scatter(c6.index, c6.values, label='Ns = 6', alpha=0.7)
    plt.scatter(c8.index, c8.values, label='Ns = 8 (aggregated)', alpha=0.7)
    plt.xlabel('Cycle Number')
    plt.ylabel('Capacity (mA.h)')
    plt.title(f'{channel}: Capacity vs. Cycle Number\n'
              f'Pearson r = {pearson_r:.4f}, R² = {r2:.4f}')
    plt.legend()
    plt.show()
    
    
    
    
    
# def gpr_weights(w):
#     # Suppose you know your frequencies (33 values)
#     freqs = np.array([0.999, 1.33, 1.78, 2.37, 3.16, 4.22, 5.62, 7.5, 
#                       10.0, 13.3, 17.8, 23.7, 31.6, 42.2, 56.2, 75.0, 
#                       102.0, 135.0, 178.0, 237.0, 316.0, 422.0, 564.0, 750.0, 
#                       1000.0, 1330.0, 1780.0, 2370.0, 3160.0, 4220.0, 5620.0, 7500.0, 
#                       10000.0])

#     # Split weights into Re and Im components
#     w_re = w[:len(freqs)]
#     w_im = w[len(freqs):]

#     # Normalize for visual comparison
#     w_re /= np.max(w_re)
#     w_im /= np.max(w_im)

#     plt.figure(figsize=(9,5))
#     plt.semilogx(freqs, w_re, 'o-', label='Re(Z) weights')
#     plt.semilogx(freqs, w_im, 's--', label='-Im(Z) weights')
#     plt.xlabel('Frequency (Hz)')
#     plt.ylabel('Relative importance weight (exp(-ℓ))')
#     plt.title('ARD Frequency Importance (Zhang-style)')
#     plt.legend()
#     plt.grid(True, which='both', ls='--', alpha=0.6)
#     plt.tight_layout()
#     plt.show()

#     # Optionally print top 5 frequencies by mean weight
#     w_mean = (w_re + w_im) / 2
#     top_idx = np.argsort(w_mean)[::-1][:5]
#     print("Top 5 most relevant frequencies (Hz):", freqs[top_idx])
#     print("Corresponding weights:", w_mean[top_idx])
    
def ard_summary(freqs_hz_ns_1, freqs_hz_ns_6, re_mean, re_std, im_mean, im_std, save_path=None):
    """
    Plot ARD frequency importance for dual Ns state vector with frequency labels.
    
    Args:
        freqs_hz_ns_1: List of frequencies for Ns=1 (discharged state)
        freqs_hz_ns_6: List of frequencies for Ns=6 (charged state)
        re_mean: Mean ARD weights for Re(Z) - concatenated [Ns1, Ns6]
        re_std: Std ARD weights for Re(Z) - concatenated [Ns1, Ns6]
        im_mean: Mean ARD weights for Im(Z) - concatenated [Ns1, Ns6]
        im_std: Std ARD weights for Im(Z) - concatenated [Ns1, Ns6]
        save_path: Optional path to save the figure
        
    Returns:
        fig: Matplotlib figure object
    """
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
    # Prepare full dataset values
    Re_Z_full = df['Re(Z)/Ohm'].values
    Im_Z_full = df['-Im(Z)/Ohm'].values

    # Prepare filtered dataset values
    filtered_df = df.loc[(df['Ns'].isin([1, 6])) & (df['cycle number'] != 0)].copy()
    Re_Z_filtered = filtered_df['Re(Z)/Ohm'].values
    Im_Z_filtered = filtered_df['-Im(Z)/Ohm'].values

    # Create subplots: two columns side by side
    fig, axes = plt.subplots(1, 2, figsize=(16, 8))

    # Plot full dataset Nyquist plot on the first subplot
    axes[0].plot(Re_Z_full, Im_Z_full, 'o', markersize=5, alpha=0.7, label="Impedance Data")
    axes[0].set_xlabel('Re(Z) / Ohm')
    axes[0].set_ylabel('Im(Z) / Ohm')
    axes[0].set_title(f'{title_prefix} Nyquist Plot of Battery Impedance')
    axes[0].grid(True)
    axes[0].axis('equal')
    axes[0].legend()

    # Plot filtered dataset Nyquist plot on the second subplot
    axes[1].plot(Re_Z_filtered, Im_Z_filtered, 'o', markersize=5, alpha=0.7, label="Impedance Data")
    axes[1].set_xlabel('Re(Z) / Ohm')
    axes[1].set_ylabel('Im(Z) / Ohm')
    axes[1].set_title(f'{title_prefix} Nyquist Plot of Battery Impedance (EIS States)')
    axes[1].grid(True)
    axes[1].axis('equal')
    axes[1].legend()

    # Adjust layout to prevent overlap
    plt.tight_layout()
    plt.show()


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

