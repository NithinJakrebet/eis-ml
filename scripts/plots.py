import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def plot_degradation(df: pd.DataFrame):
      capacity = df['Capacity/mA.h']
      cycle_number = df['cycle number']
      
      plt.figure(figsize=(8, 8))
      plt.scatter(cycle_number, capacity, alpha=0.6)
      plt.xlabel('Cycle Number')
      plt.ylabel('Capacity (mA.h)')
      plt.title('Capacity vs. Cycle Number')
      plt.show()

      
      
def plot_unique_frequencies(unique_freqs: np.array):
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

      
def plot_nyquist(df: pd.DataFrame):
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

