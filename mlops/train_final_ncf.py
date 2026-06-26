"""
Final Production Training Script for Neural Collaborative Filtering (NCF).

This script trains the winning NCF deep learning architecture on the fully 
processed interaction dataset, logs the training hyperparameters and resulting 
evaluation metrics to the MLflow server under production experiments, and 
serializes the final PyTorch model state dictionary.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
import pandas as pd
import numpy as np
import mlflow
import yaml

# Safely resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.ncf_model import NeuralCFRecommender, NCFDataset
from mlops.train import prepare_sparse_matrices
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker


def run_final_ncf_production():
    """Executes the end-to-end production pipeline for training and logging NCF."""
    # 1. Load Configurations
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'data_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    data_dir = os.path.join(PROJECT_ROOT, data_config['paths']['processed_data'])
    models_dir = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(models_dir, exist_ok=True)
    
    # 2. Load Datasets
    print("[INFO] Loading datasets for final NCF production training...")
    train_df = pd.read_parquet(os.path.join(data_dir, data_config['paths']['train_file']))
    val_df = pd.read_parquet(os.path.join(data_dir, data_config['paths']['val_file']))
    
    # Filter 2017+ to maintain consistency across global pipeline logic
    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    
    # 3. Prepare Sparse Matrices
    print("[INFO] Preparing sparse matrices...")
    train_matrix, val_matrix, eval_users = prepare_sparse_matrices(train_df, val_df)
    num_users, num_items = train_matrix.shape
    
    # 4. Set Winning Hyperparameters
    final_ncf_params = {
        'num_users': num_users,
        'num_items': num_items,
        'emb_dim': 32,
        'epochs': 5,
        'batch_size': 2048,
        'learning_rate': 0.001,
        'num_negatives': 4
    }
    
    # 5. Initialize PyTorch Infrastructure
    print("[INFO] Initializing PyTorch Dataset and DataLoader...")
    train_dataset = NCFDataset(
        train_matrix=train_matrix, 
        num_items=num_items, 
        num_negatives=final_ncf_params['num_negatives']
    )
    train_loader = DataLoader(
        train_dataset, 
        batch_size=final_ncf_params['batch_size'], 
        shuffle=True, 
        num_workers=0
    )
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ncf_recommender = NeuralCFRecommender(
        num_users=num_users, 
        num_items=num_items, 
        emb_dim=final_ncf_params['emb_dim']
    )
    ncf_recommender.fit(train_matrix)
    
    model = ncf_recommender.model
    optimizer = optim.Adam(model.parameters(), lr=final_ncf_params['learning_rate'])
    criterion = nn.BCELoss()
    
    # 6. Setup MLflow Tracking Context
    tracker = MLFlowTracker(experiment_name="3_Production_Models")
    
    print("\n[INFO] Starting Final NCF Production Optimization Loop...")
    with tracker.start_run(run_name="Neural_CF_Final_Model"):
        # Log deep learning hyper-parameters
        mlflow.log_params(final_ncf_params)
        
        # Explicit Training Loop
        for epoch in range(1, final_ncf_params['epochs'] + 1):
            model.train()
            total_loss = 0.0
            
            for batch_users, batch_items, batch_labels in train_loader:
                batch_users = batch_users.to(device).long()
                batch_items = batch_items.to(device).long()
                batch_labels = batch_labels.to(device)
                
                optimizer.zero_grad()
                predictions = model(batch_users, batch_items)
                loss = criterion(predictions, batch_labels)
                loss.backward()
                optimizer.step()
                
                total_loss += loss.item()
                
            avg_loss = total_loss / len(train_loader)
            mlflow.log_metric("train_loss_epoch", avg_loss, step=epoch)
            print(f"[TRAIN LOG] Epoch {epoch}/{final_ncf_params['epochs']} | Loss: {avg_loss:.4f}")
            
        # 7. Model Evaluation Pipeline
        print("\n[INFO] Running evaluation pipeline on validation set...")
        metrics = evaluate_model_at_k(ncf_recommender, val_matrix, eval_users, k=10)
        mlflow.log_metrics(metrics)
        print(f"[INFO] Final NCF Validation NDCG@10: {metrics.get('NDCG_10', 0):.4f}")
        
        # 8. Save Model Weights Locally and Log to Server
        model_path = os.path.join(models_dir, 'best_ncf_model.pt')
        torch.save(model.state_dict(), model_path)
        print(f"[INFO] Model weights successfully saved locally to: {model_path}")
        
        mlflow.log_artifact(model_path, artifact_path="model_artifact")
        print("[INFO] Model artifact successfully registered to MLflow tracking storage.")


if __name__ == "__main__":
    run_final_ncf_production()