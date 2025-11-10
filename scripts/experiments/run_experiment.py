import sys
import yaml
import argparse
from pathlib import Path
import joblib

project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root / "scripts"))
from data_pipeline import load_and_prepare_data
from algorithms import gpr, xgb
import evaluate


def get_project_root(): 
      return Path(__file__).resolve().parent.parent.parent

def run_experiment(algorithm, experiment_name, save_model_path=None):
      project_root = get_project_root()
      config_path = project_root / f"configs/{algorithm}.yaml"
    
      if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)
      
      with open(config_path, 'r') as f:
            config = yaml.safe_load(f)
      
      if experiment_name not in config:
            print(f"Error: Experiment '{experiment_name}' not found in {algorithm}.yaml")
            print(f"Available experiments: {list(config.keys())}")
            sys.exit(1)
      
      experiment_config = config[experiment_name]

      data_folder = project_root / "data" / experiment_config["data_folder"]
      split_method = experiment_config.get('split_method', 'temporal')
      
      print(f"Running: {algorithm} / {experiment_name}")
      print(f"Data: {experiment_config['data_folder']}")
      print(f"Split: {split_method}")
      
      X_train, y_train, X_test, y_test = load_and_prepare_data(data_folder=data_folder, method=split_method)
      
      print(f"Train: {X_train.shape[0]} samples, Test: {X_test.shape[0]} samples")
      
      if algorithm == "gpr":
            model = gpr.train_capacity_gpr_fast(
                  X_train, y_train,
                  **experiment_config['model_params'],
                  kernel_params=experiment_config['kernel_params'],
                  gpr_params=experiment_config['gpr_params']
            )
            y_pred, y_std = gpr.predict_fast(model, X_test)
      elif algorithm == "xgb":
            model = xgb.train_ensemble_model(X_train, y_train)
            y_pred, std_predictions, all_predictions= xgb.predict_ensemble(model, X_test)
            
            
      else:
            print(f"Error: Algorithm '{algorithm}' not yet implemented")
            sys.exit(1)
      
      
      rmse, r2, mse, mae = evaluate.evaluate_model(y_test, y_pred)
      
      print(f"\nResults:")
      print(f"  RMSE: {rmse:.4f}")
      print(f"  MAE:  {mae:.4f}")
      print(f"  R²:   {r2:.4f}")
      
      if save_model_path:

            if not Path(save_model_path).is_absolute(): save_model_path = project_root / save_model_path
            else: save_model_path = Path(save_model_path)

            save_model_path.parent.mkdir(parents=True, exist_ok=True)

            joblib.dump(model, save_model_path)
            print(f"\nModel saved to: {save_model_path}")
      
      return model, (rmse, r2, mae)









def main():
      parser = argparse.ArgumentParser(description="Run ML experiments with YAML configurations")
      parser.add_argument("algorithm", choices=["gpr", "xgb"], help="Algorithm to use")
      parser.add_argument("experiment", help="Experiment name from config YAML")
      parser.add_argument("--save-model", help="Path to save trained model")
      
      args = parser.parse_args()
      run_experiment(args.algorithm, args.experiment, args.save_model)
      
if __name__ == '__main__':
    main()