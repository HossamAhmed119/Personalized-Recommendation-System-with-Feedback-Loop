import os
import optuna
import pandas as pd

# Fix OpenBLAS warning by disabling its internal threadpool
os.environ['OPENBLAS_NUM_THREADS'] = '1'

# -------------------------------------------------------------------
# Imports from project structure
# -------------------------------------------------------------------
from src.utils.config_loader import load_tuning_config
from mlops.train import train_als
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker

def objective(trial, config):
    """
    Optuna objective function, driven by the YAML configuration.
    """
    # 1. Read search space dynamically from the loaded config
    space = config["search_space"]
    
    factors = trial.suggest_categorical(
        "factors", 
        space["factors"]["choices"]
    )
    iterations = trial.suggest_int(
        "iterations", 
        space["iterations"]["low"], 
        space["iterations"]["high"], 
        step=space["iterations"]["step"]
    )
    regularization = trial.suggest_float(
        "regularization", 
        space["regularization"]["low"], 
        space["regularization"]["high"], 
        log=True
    )

    # 2. Initialize tracker within the trial using config values
    tracker = MLFlowTracker(
        experiment_name=config["experiment"]["name"], 
        db_uri=config["experiment"]["mlflow_uri"]
    )

    # Dynamically name the run using the Optuna trial number for better dashboard readability
    run_name = f"Trial_{trial.number:02d}"

    # 3. Execute the trial within a nested MLflow run
    with tracker.start_run(run_name=run_name, nested=True):
        
        tracker.log_params({
            "factors": factors,
            "iterations": iterations,
            "regularization": regularization
        })

        # Load pre-split datasets
        train_df = pd.read_parquet(config["experiment"]["train_path"])
        val_df = pd.read_parquet(config["experiment"]["val_path"])
        
        # Train the model using the parameters for this trial
        # Unpack the 4 returned variables to ensure correct evaluation filtering
        model, train_matrix, val_matrix, test_users = train_als(
            train_df=train_df,
            val_df=val_df,
            factors=factors,
            iterations=iterations,
            regularization=regularization
        )

        # Evaluate the trained model
        metrics_dict = evaluate_model_at_k(
            model=model, 
            train_matrix=train_matrix,
            val_matrix=val_matrix, 
            users_to_evaluate=test_users, 
            k=10
        )
        
        # Log all calculated metrics (HitRate_10, Precision_10, Recall_10, MRR_10, NDCG_10)
        tracker.log_metrics(metrics_dict)
        
        # Optuna will attempt to maximize this specific metric
        return metrics_dict['HitRate_10']

def main():
    """
    Main entry point for the hyperparameter tuning script.
    """
    # Define the path to the specific experiment configuration
    CONFIG_PATH = "configs/experiments/exp_001.yaml"
    
    # Use the specific tuning config loader with validation
    config = load_tuning_config(CONFIG_PATH)

    print(f"Starting Optuna Tuning for Experiment: {config['experiment']['name']}")
    
    # Initialize the Optuna study with its dedicated URI to avoid database conflicts
    study = optuna.create_study(
        direction="maximize",
        study_name=config["experiment"]["name"],
        storage=config["experiment"]["optuna_uri"], 
        load_if_exists=True
    )
    
    # Execute the optimization loop
    study.optimize(
        lambda trial: objective(trial, config), 
        n_trials=config["experiment"]["n_trials"]
    )

    # Output final results to the console
    print("\nBest Trial Results:")
    print(f"  Best HitRate_10 : {study.best_value:.4f}")
    print(f"  Best Params     : {study.best_params}")

if __name__ == "__main__":
    main()