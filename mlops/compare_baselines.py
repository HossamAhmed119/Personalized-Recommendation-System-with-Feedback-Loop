# File: mlops/compare_baselines.py
import os
import sys
import time
import pandas as pd
import mlflow

# Resolve project root 
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.utils.config_loader import load_model_config
from src.models.cf_model import get_model, get_model_params
from mlops.train import prepare_sparse_matrices
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker

def run_baseline_comparison():
    # 1. Paths and Configs
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'model_config.yaml')
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    
    model_configs = load_model_config(CONFIG_PATH)
    
    # 2. Load and Filter Data
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val.parquet'))
    
    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    print(f"Filtered Train size (2017+): {train_df.shape[0]:,} interactions")
    
    print("Preparing sparse matrices...")
    train_matrix, val_matrix, eval_users = prepare_sparse_matrices(train_df, val_df)
    
    # 3. Setup MLflow Tracker
    tracker = MLFlowTracker(experiment_name="1_Baseline_Models_Comparison")
    
    models_to_compare = ['ALS', 'BPR', 'SVD']
    
    print("\nStarting Baseline Comparisons in MLflow...")
    for model_name in models_to_compare:
        print(f"\n{'='*40}\nRunning {model_name}...\n{'='*40}")
        
        params = get_model_params(model_configs, model_name)
        model = get_model(model_name, params)
        
        with tracker.start_run(run_name=f"{model_name}_Baseline"):
            mlflow.log_param("model_type", model_name)
            mlflow.log_params(params)
            
            start_train = time.time()
            model.fit(train_matrix)
            train_time = time.time() - start_train
            mlflow.log_metric("Train_Time_sec", train_time)
            
            start_eval = time.time()
            metrics = evaluate_model_at_k(model, val_matrix, eval_users, k=10)
            eval_time = time.time() - start_eval
            mlflow.log_metric("Eval_Time_sec", eval_time)
            
            mlflow.log_metrics(metrics)
            print(f" {model_name} logged successfully! NDCG@10: {metrics.get('NDCG_10', 0)}")

if __name__ == "__main__":
    run_baseline_comparison()