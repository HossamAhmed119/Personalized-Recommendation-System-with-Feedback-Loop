# File: mlops/tune.py
import os
import sys
import optuna
import pandas as pd
import mlflow

# Resolve project root cls
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from mlops.train import prepare_sparse_matrices
from src.models.cf_model import ALSRecommender
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker

def run_tuning():
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val.parquet'))
    
    # Filter 2017+
    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    
    print("Preparing sparse matrices...")
    train_matrix, val_matrix, eval_users = prepare_sparse_matrices(train_df, val_df)
    
    # Setup MLflow Tracker 
    tracker = MLFlowTracker(experiment_name="2_ALS_Hyperparameter_Tuning")
    
    def objective(trial):
        factors = trial.suggest_categorical('factors', [32, 64, 100, 128])
        regularization = trial.suggest_float('regularization', 0.01, 0.1, log=True)
        iterations = trial.suggest_int('iterations', 10, 30, step=5)
        
        with tracker.start_run(run_name=f"Trial_{trial.number}", nested=True):
            params = {
                'factors': factors,
                'regularization': regularization,
                'iterations': iterations,
                'random_state': 42
            }
            
            mlflow.log_params(params)
            
            # Train
            model = ALSRecommender(**params)
            model.fit(train_matrix)
            
            # Evaluate
            metrics = evaluate_model_at_k(model, val_matrix, eval_users, k=10)
            mlflow.log_metrics(metrics)
            
            # Optimize for NDCG
            return metrics.get('NDCG_10', 0)

    print("\nStarting Optuna Sweep...")
    with tracker.start_run(run_name="ALS_Optuna_Parent_Run"):
        study = optuna.create_study(direction="maximize")
        study.optimize(objective, n_trials=15) 
        
        print("\n Best Trial Found:")
        print(f"  NDCG@10: {study.best_trial.value}")
        print(f"  Params: {study.best_trial.params}")
        
        mlflow.log_params({"best_" + k: v for k, v in study.best_trial.params.items()})

if __name__ == "__main__":
    run_tuning()