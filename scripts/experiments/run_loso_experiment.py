"""
LOSO (Leave-One-Subject-Out) Cross-Validation Experiment Runner

This script runs LOSO cross-validation experiments using configurations from gpr.yaml.
Each cell is used once as test set while all other cells form the training set.

Usage:
    python run_loso_experiment.py PEIS-HC-RT_LOSO
"""

import sys
import yaml
import argparse
import json
import numpy as np
import pandas as pd
from pathlib import Path
import joblib
import warnings
warnings.filterwarnings('ignore')

# Add project scripts to path
project_root = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(project_root / "scripts"))

from data_pipeline import loso
from feature_engineering import build_model_input
from algorithms import gpr
import plots
import evaluate


def get_project_root():
    return Path(__file__).resolve().parent.parent.parent


def run_loso_experiment(experiment_name):
    project_root = get_project_root()
    config_path = project_root / "configs/gpr.yaml"
    
    if not config_path.exists():
        print(f"Error: Config file not found: {config_path}")
        sys.exit(1)
    
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    if experiment_name not in config:
        print(f"Error: Experiment '{experiment_name}' not found in gpr.yaml")
        print(f"Available experiments: {list(config.keys())}")
        sys.exit(1)
    
    exp_config = config[experiment_name]
    
    if exp_config.get('experiment_type') != 'loso_cross_validation':
        print(f"Error: Experiment '{experiment_name}' is not configured for LOSO")
        sys.exit(1)
    
    # Setup paths
    data_path = project_root / "data" / exp_config["data_folder"]
    model_dir = project_root / exp_config["paths"]["model_dir"]
    results_dir = project_root / exp_config["paths"]["results_dir"]
    plots_dir = project_root / exp_config["paths"]["plots_dir"]
    
    model_dir.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)
    
    channels = exp_config["channels"]
    
    print("="*70)
    print(f"Running LOSO Cross-Validation: {experiment_name}")
    print("="*70)
    print(f"Data: {exp_config['data_folder']}")
    print(f"Channels: {len(channels)} cells")
    print(f"Model dir: {model_dir.relative_to(project_root)}")
    print(f"Results dir: {results_dir.relative_to(project_root)}")
    print("="*70)
    
    # Training loop
    results = []
    W_re, W_im = [], []
    
    for i, test_cell in enumerate(channels):
        print(f"\n[{i+1}/{len(channels)}] Training with {test_cell} as test set...")
        
        train_cells = [c for c in channels if c != test_cell]
        df_train, df_test = loso(data_path, train_cells, [test_cell])
        
        X_train, y_train = build_model_input(df_train)
        X_test, y_test = build_model_input(df_test)
        
        print(f"  Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
        
        # Train model
        model = gpr.train_capacity_gpr_fast(
            X_train, 
            y_train,
            **exp_config['model_params'],
            kernel_params=exp_config['kernel_params'],
            gpr_params=exp_config['gpr_params']
        )
        
        # Save model
        model_path = model_dir / f"{test_cell}.pkl"
        joblib.dump(model, model_path)
        
        # Predict
        y_pred, y_std = gpr.predict_fast(model, X_test)
        rmse, r2, mse, mae = evaluate.evaluate_model(y_test, y_pred)
        results.append((test_cell, rmse, mae, r2))
        
        print(f"  Results: RMSE={rmse:.4f}, R²={r2:.4f}, MAE={mae:.4f}")
        
        w = gpr.ard_frequency_weights(model)

        W_re.append(np.concatenate([w[0:37], w[74:107]]))    # Re from Ns1 and Ns6
        W_im.append(np.concatenate([w[37:74], w[107:140]]))  # Im from Ns1 and Ns6
    
    print(f"\n{'='*70}")
    print(f"Training complete! {len(results)} models saved.")
    print("="*70)
    
    # Save results
    df = pd.DataFrame(results, columns=["cell", "rmse", "mae", "r2"]).sort_values("cell")
    summary = pd.DataFrame({
        "cell": ["_mean_"],
        "rmse": [df["rmse"].mean()],
        "mae": [df["mae"].mean()],
        "r2": [df["r2"].mean()],
    })
    df_out = pd.concat([df, summary], ignore_index=True)
    
    csv_path = results_dir / f"gpr_{len(channels)}_fold_cv_results.csv"
    json_path = results_dir / f"gpr_{len(channels)}_fold_cv_results.json"
    
    df_out.to_csv(csv_path, index=False)
    with open(json_path, "w") as f:
        json.dump(df.to_dict(orient="records"), f, indent=2)
    
    print(f"\nResults saved to:")
    print(f"  {csv_path.relative_to(project_root)}")
    print(f"  {json_path.relative_to(project_root)}")
    
    print("\nPerformance Summary:")
    print(df_out.to_string(index=False))
    
    # Process ARD weights
    print("\nProcessing ARD weights...")
    W_re = np.vstack(W_re)
    W_im = np.vstack(W_im)
    
    re_mean, re_std = W_re.mean(axis=0), W_re.std(axis=0)
    im_mean, im_std = W_im.mean(axis=0), W_im.std(axis=0)
    avg_mean = (re_mean + im_mean) / 2.0
    avg_std = np.sqrt((re_std**2 + im_std**2) / 2.0)
    
    # Get frequencies from config
    freqs_hz_ns_1 = exp_config['frequencies']['ns1']
    freqs_hz_ns_6 = exp_config['frequencies']['ns6']
    
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
    ard_csv_path = results_dir / "ard_weights_across_folds.csv"
    ard_df.to_csv(ard_csv_path, index=False)
    
    print(f"\nARD weights saved to: {ard_csv_path.relative_to(project_root)}")
    
    # Generate ARD plot
    print("\nGenerating ARD weight plot...")
    png_path = results_dir / "ard_weights_across_folds.png"
    fig = plots.ard_summary(
        freqs_hz_ns_1, 
        freqs_hz_ns_6, 
        re_mean, 
        re_std, 
        im_mean, 
        im_std, 
        save_path=png_path
    )
    print(f"ARD plot saved to: {png_path.relative_to(project_root)}")
    
    # Generate diagnostic plots for all cells
    print("\nGenerating diagnostic plots for all cells...")
    for test_cell in channels:
        # Load model and data
        model_path = model_dir / f"{test_cell}.pkl"
        model = joblib.load(model_path)
        
        train_cells = [c for c in channels if c != test_cell]
        _, df_test = loso(data_path, train_cells, [test_cell])
        
        X_test, y_test = build_model_input(df_test)
        y_pred, y_std = gpr.predict_fast(model, X_test)
        
        # Get cycle numbers
        test_cycles = df_test[df_test['Ns'] == 1].groupby('cycle number').first().index.values
        
        # Plot 1: Model predictions
        fig1 = plots.model_predictions(y_test, y_pred, y_std, title_prefix=f"{test_cell}")
        fig1.savefig(plots_dir / f"{test_cell}_predictions.png", dpi=150, bbox_inches='tight')
        
        # Plot 2: Residual analysis
        residuals = y_test - y_pred
        fig2 = plots.residual_analysis_plots(residuals, test_cycles, y_pred, y_true=y_test)
        fig2.savefig(plots_dir / f"{test_cell}_residuals.png", dpi=150, bbox_inches='tight')
        
        # Plot 3: Capacity vs cycle
        fig3 = plots.capacity_vs_cycle(y_test, y_pred, test_cycles, y_std, title_prefix=f"{test_cell}")
        fig3.savefig(plots_dir / f"{test_cell}_capacity_vs_cycle.png", dpi=150, bbox_inches='tight')
        
        print(f"  {test_cell}: 3 plots saved")
    
    print(f"\nAll diagnostic plots saved to: {plots_dir.relative_to(project_root)}")
    print(f"Total plots generated: {len(channels) * 3} (3 per cell)")
    
    print("\n" + "="*70)
    print("LOSO Experiment Complete!")
    print("="*70)


def main():
    parser = argparse.ArgumentParser(
        description="Run LOSO Cross-Validation experiments with YAML configurations"
    )
    parser.add_argument(
        "experiment", 
        help="Experiment name from gpr.yaml (e.g., PEIS-HC-RT_LOSO)"
    )
    
    args = parser.parse_args()
    run_loso_experiment(args.experiment)


if __name__ == '__main__':
    main()
