import matplotlib.pyplot as plt
plt.style.use("paper.mplstyle")



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

    
    
    
    
    
def gpr_weights(w):
    # Suppose you know your frequencies (33 values)
    freqs = np.array([0.999, 1.33, 1.78, 2.37, 3.16, 4.22, 5.62, 7.5, 
                      10.0, 13.3, 17.8, 23.7, 31.6, 42.2, 56.2, 75.0, 
                      102.0, 135.0, 178.0, 237.0, 316.0, 422.0, 564.0, 750.0, 
                      1000.0, 1330.0, 1780.0, 2370.0, 3160.0, 4220.0, 5620.0, 7500.0, 
                      10000.0])

    # Split weights into Re and Im components
    w_re = w[:len(freqs)]
    w_im = w[len(freqs):]

    # Normalize for visual comparison
    w_re /= np.max(w_re)
    w_im /= np.max(w_im)

    plt.figure(figsize=(9,5))
    plt.semilogx(freqs, w_re, 'o-', label='Re(Z) weights')
    plt.semilogx(freqs, w_im, 's--', label='-Im(Z) weights')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Relative importance weight (exp(-ℓ))')
    plt.title('ARD Frequency Importance (Zhang-style)')
    plt.legend()
    plt.grid(True, which='both', ls='--', alpha=0.6)
    plt.tight_layout()
    plt.show()

    # Optionally print top 5 frequencies by mean weight
    w_mean = (w_re + w_im) / 2
    top_idx = np.argsort(w_mean)[::-1][:5]
    print("Top 5 most relevant frequencies (Hz):", freqs[top_idx])
    print("Corresponding weights:", w_mean[top_idx])
    
def ard_summary(freqs_hz, re_mean, re_std, im_mean, im_std, avg_mean):
    plt.figure(figsize=(10,5))
    plt.semilogx(freqs_hz, re_mean, 'o-', label='Re(Z) mean')
    plt.fill_between(freqs_hz, re_mean-re_std, re_mean+re_std, alpha=0.2)
    plt.semilogx(freqs_hz, im_mean, 's--', label='-Im(Z) mean')
    plt.fill_between(freqs_hz, im_mean-im_std, im_mean+im_std, alpha=0.2)
    plt.semilogx(freqs_hz, avg_mean, '-', linewidth=2, label='Avg(Re,Im) mean')
    plt.xlabel("Frequency (Hz)"); plt.ylabel("Relative importance (exp(-ℓ))")
    plt.title("ARD frequency importance across 8 LOSO folds")
    plt.grid(True, which='both', ls='--', alpha=0.4)
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    return plt
    
    
    
    
    
    

      
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



def residual_analysis_plots(residuals, cycle_numbers, y_pred_mean):
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

