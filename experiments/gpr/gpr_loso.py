
import numpy as np
import pandas as pd
import warnings
import sys
from pathlib import Path
warnings.filterwarnings("ignore")

sys.path.append("scripts")  

from data_pipeline import loso
from feature_engineering import build_model_input
from algorithms import gpr
import evaluate

data_folder = "PEIS-HC-RT"
cv_method = "LOSO"

# Model parameters
model_params = {
    'subset_size': 300,
    'top_k_freqs': None
}

# Kernel parameters
kernel_params = {
    'constant_value': 1.0,
    'constant_bounds': [0.01, 100.0],
    'rbf_length_scale_bounds': [0.01, 100.0],
    'white_noise_level': 0.001,
    'white_noise_bounds': 'fixed'
}

# GPR parameters
gpr_params = {
    'alpha': 0.0,
    'normalize_y': True,
    'n_restarts_optimizer': 1,
    'random_state': 42
}

print(f"Configuration loaded:")
print(f"  Data: {data_folder}")
print(f"  CV Method: {cv_method}")

CHANNELS = [
      'A1','A2','A3','A4','A5','A6','A7','A8',
      'B1','B2','B3','B4','B5','B6'
]

results = []
models = {}
W_re, W_im = [], []

data_path = Path("data") / data_folder

print("Starting training loop")
# took me around 80 minutes - Nithin
for i, test_cell in enumerate(CHANNELS):
    train_cells = [c for c in CHANNELS if c != test_cell]
    df_train, df_test = loso(data_path, train_cells, [test_cell])
    
    X_train, y_train = build_model_input(df_train)
    X_test, y_test = build_model_input(df_test)
    
    model = gpr.train_capacity_gpr_fast(
      X_train, 
      y_train,
      **model_params,
      kernel_params=kernel_params,
      gpr_params=gpr_params
    )
    
    models[test_cell] = model
    
    y_pred, y_std = gpr.predict_fast(model, X_test)
    rmse, r2, mse, mae = evaluate.evaluate_model(y_test, y_pred)
    results.append((test_cell, rmse, mae, r2))
    
    # Extract ARD weights for dual Ns state vector
    w = gpr.ard_frequency_weights(model)
    # Structure: [Re(Z)_Ns1 (37), Im(Z)_Ns1 (37), Re(Z)_Ns6 (33), Im(Z)_Ns6 (33)]
    # Total: 140 features
    W_re.append(np.concatenate([w[0:37], w[74:107]]))    # Re from Ns1 and Ns6
    W_im.append(np.concatenate([w[37:74], w[107:140]]))  # Im from Ns1 and Ns6

print(f"Completed {len(results)} folds")

df = pd.DataFrame(results, columns=["cell", "rmse", "mae", "r2"]).sort_values("cell")
summary = pd.DataFrame({
    "cell": ["_mean_"],
    "rmse": [df["rmse"].mean()],
    "mae": [df["mae"].mean()],
    "r2": [df["r2"].mean()],
})
df_out = pd.concat([df, summary], ignore_index=True)

print("\nPerformance Summary:")
print(df_out)

W_re = np.vstack(W_re)  # Shape: (n_folds, 70) = 37 from Ns1 + 33 from Ns6
W_im = np.vstack(W_im)  # Shape: (n_folds, 70) = 37 from Ns1 + 33 from Ns6

re_mean, re_std = W_re.mean(axis=0), W_re.std(axis=0)
im_mean, im_std = W_im.mean(axis=0), W_im.std(axis=0)
avg_mean = (re_mean + im_mean) / 2.0
avg_std = np.sqrt((re_std**2 + im_std**2) / 2.0)

# Frequencies for Ns=1 (37 freqs) - discharged state
freqs_hz_ns_1 = [
    0.254, 0.34, 0.456, 0.612, 0.822, 1.1, 1.48, 1.99, 
    2.66, 3.57, 4.8, 6.43, 8.64, 11.6, 15.5, 
    20.9, 28.0, 37.5, 50.3, 67.6, 90.6, 122.0, 163.0, 
    219.0, 294.0, 394.0, 529.0, 710.0, 952.0, 1280.0, 1710.0, 
    2300.0, 3090.0, 4140.0, 5560.0, 7450.0, 10000.0
]

# Frequencies for Ns=6 (33 freqs) - charged state
freqs_hz_ns_6 = [
    0.999, 1.33, 1.78, 2.37, 3.16, 4.22, 5.62, 7.5, 10.0, 13.3, 17.8, 23.7,
    31.6, 42.2, 56.2, 75.0, 102.0, 135.0, 178.0, 237.0, 316.0, 422.0, 564.0,
    750.0, 1000.0, 1330.0, 1780.0, 2370.0, 3160.0, 4220.0, 5620.0, 7500.0, 10000.0
]

# Combine frequencies and weights
all_freqs = freqs_hz_ns_1 + freqs_hz_ns_6
ns_labels = ['Ns1'] * len(freqs_hz_ns_1) + ['Ns6'] * len(freqs_hz_ns_6)

# Save ARD weights
ard_df = pd.DataFrame({
    "ns_state": ns_labels,
    "freq_hz": all_freqs,
    "w_re_mean": re_mean, 
    "w_re_std": re_std,
    "w_im_mean": im_mean, 
    "w_im_std": im_std,
    "w_avg_mean": avg_mean, 
    "w_avg_std": avg_std,
})

print(f"Total frequencies: {len(all_freqs)} (37 from Ns1 + 33 from Ns6)")
print(f"Weight array shape: Re={W_re.shape}, Im={W_im.shape}")

# Extract data from DataFrame
freqs_ns1 = ard_df[ard_df['ns_state'] == 'Ns1']['freq_hz'].values
freqs_ns6 = ard_df[ard_df['ns_state'] == 'Ns6']['freq_hz'].values

re_mean = ard_df['w_re_mean'].values
re_std = ard_df['w_re_std'].values
im_mean = ard_df['w_im_mean'].values
im_std = ard_df['w_im_std'].values

re_mean_ns1 = ard_df[ard_df['ns_state'] == 'Ns1']['w_re_mean'].values
re_std_ns1 = ard_df[ard_df['ns_state'] == 'Ns1']['w_re_std'].values
im_mean_ns1 = ard_df[ard_df['ns_state'] == 'Ns1']['w_im_mean'].values
im_std_ns1 = ard_df[ard_df['ns_state'] == 'Ns1']['w_im_std'].values

re_mean_ns6 = ard_df[ard_df['ns_state'] == 'Ns6']['w_re_mean'].values
re_std_ns6 = ard_df[ard_df['ns_state'] == 'Ns6']['w_re_std'].values
im_mean_ns6 = ard_df[ard_df['ns_state'] == 'Ns6']['w_im_mean'].values
im_std_ns6 = ard_df[ard_df['ns_state'] == 'Ns6']['w_im_std'].values

re_mean_combined = np.concatenate([re_mean_ns1, re_mean_ns6])
re_std_combined = np.concatenate([re_std_ns1, re_std_ns6])
im_mean_combined = np.concatenate([im_mean_ns1, im_mean_ns6])
im_std_combined = np.concatenate([im_std_ns1, im_std_ns6])


print(f"  Total frequencies: {len(freqs_ns1)} (Ns1) + {len(freqs_ns6)} (Ns6) = {len(freqs_ns1) + len(freqs_ns6)}")

weights_summary = pd.DataFrame({
    'Ns State': ard_df['ns_state'],
    'Frequency (Hz)': ard_df['freq_hz'],
    'Re(Z) Weight': ard_df['w_re_mean'],
    'Im(Z) Weight': ard_df['w_im_mean'],
    'Avg Weight': ard_df['w_avg_mean']
})

# Sort by average weight (descending)
weights_summary = weights_summary.sort_values('Avg Weight', ascending=False).reset_index(drop=True)

print("ARD Frequency Weights (Sorted by Importance)")
print("=" * 80)
print(f"{'Rank':<6} {'Ns':<5} {'Freq (Hz)':<12} {'Re(Z) Wt':<12} {'Im(Z) Wt':<12} {'Avg Wt':<10}")
print("-" * 80)

for idx, row in weights_summary.iterrows():
    print(f"{idx+1:<6} {row['Ns State']:<5} {row['Frequency (Hz)']:<12.2f} "
          f"{row['Re(Z) Weight']:<12.4f} {row['Im(Z) Weight']:<12.4f} {row['Avg Weight']:<10.4f}")