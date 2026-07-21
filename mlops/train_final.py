# File: mlops/train_final.py
import os
import sys
import pickle
import pandas as pd
import mlflow

# Resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from mlops.train import prepare_sparse_matrices
from src.models.cf_model import ALSRecommender
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker

def run_final_production():
    DATA_DIR = os.path.join(PROJECT_ROOT, 'data', 'processed')
    REPORTS_DIR = os.path.join(PROJECT_ROOT, 'reports')
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    print("Loading datasets...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, 'train.parquet'))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, 'val.parquet'))
    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    
    train_matrix, val_matrix, eval_users = prepare_sparse_matrices(train_df, val_df)
    
    best_params = {
        'factors': 32,             
        'regularization': 0.05943965250604479,    
        'iterations': 30,          
        'random_state': 42
    }
    
    # Setup MLflow Tracker 
    tracker = MLFlowTracker(experiment_name="3_Production_Models")
    
    print("\nStarting Final Production Run...")
    with tracker.start_run(run_name="ALS_Final_Model"):
        mlflow.log_params(best_params)
        
        print("Training final model...")
        model = ALSRecommender(**best_params)
        model.fit(train_matrix)
        
        print("Evaluating final model...")
        metrics = evaluate_model_at_k(model, val_matrix, eval_users, k=10)
        mlflow.log_metrics(metrics)
        print(f"Final NDCG@10: {metrics.get('NDCG_10', 0)}")
        
        # Save Model
        model_path = os.path.join(MODELS_DIR, 'best_als_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"Model saved to {model_path}")
        
        # Log Artifacts
        mlflow.log_artifact(model_path, artifact_path="model_artifact")
        
        plots_to_log = ['model_comparison.png', 'model_time_comparison.png', 'model_comparison_results.csv']
        for plot in plots_to_log:
            file_path = os.path.join(REPORTS_DIR, plot)
            if os.path.exists(file_path):
                mlflow.log_artifact(file_path, artifact_path="evaluation_reports")
                
        print(" Production pipeline completed! Check MLflow UI.")

if __name__ == "__main__":
    run_final_production()