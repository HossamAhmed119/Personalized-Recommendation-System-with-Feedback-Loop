# File: mlops/train_final_cb.py
import os
import sys
import pickle
import pandas as pd
import mlflow
import yaml

# Safely resolve project root
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.cb_model import ContentBasedRecommender
from mlops.train import prepare_sparse_matrices
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker
from src.data_pipeline.preprocess import clean_metadata_text

def run_final_cb_production():
    # 1. Load Configurations
    CONFIG_PATH = os.path.join(PROJECT_ROOT, 'configs', 'data_config.yaml')
    with open(CONFIG_PATH, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    DATA_DIR = os.path.join(PROJECT_ROOT, data_config['paths']['processed_data'])
    MODELS_DIR = os.path.join(PROJECT_ROOT, 'models')
    os.makedirs(MODELS_DIR, exist_ok=True)
    
    # 2. Load Datasets
    print("[INFO] Loading datasets for final production training...")
    train_df = pd.read_parquet(os.path.join(DATA_DIR, data_config['paths']['train_file']))
    val_df = pd.read_parquet(os.path.join(DATA_DIR, data_config['paths']['val_file']))
    meta_df = pd.read_parquet(os.path.join(DATA_DIR, data_config['paths']['metadata_file']))
    
    # Filter 2017+ 
    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    
    # 3. Align Metadata with Interaction Data Indices
    print("[INFO] Aligning Metadata with Interaction Data...")
    all_interactions = pd.concat([train_df, val_df])
    cf_items_encoded = all_interactions['parent_asin'].unique()
    
    mapping_df = all_interactions.drop_duplicates('parent_asin')
    encoded_to_original = dict(zip(mapping_df['parent_asin'], mapping_df['parent_asin_original']))
    cf_items_original = [encoded_to_original[encoded] for encoded in cf_items_encoded]
    
    meta_df_aligned = meta_df.set_index('parent_asin').loc[cf_items_original].reset_index()
    
    # 4. Clean and Combine Text Features
    print("[INFO] Cleaning text features...")
    text_cols = ['title', 'categories', 'features', 'description']
    for col in text_cols:
        meta_df_aligned[col] = meta_df_aligned[col].apply(clean_metadata_text)
        
    meta_df_aligned['combined_text'] = (
        meta_df_aligned['title'] + " " +
        meta_df_aligned['categories'] + " " +
        meta_df_aligned['features'] + " " +
        meta_df_aligned['description']
    ).str.lower()

    # 5. Prepare Sparse Matrices
    print("[INFO] Preparing sparse matrices...")
    train_matrix, val_matrix, eval_users = prepare_sparse_matrices(train_df, val_df)
    
    # 6. Set Final Hyperparameters (The ones that gave NDCG: 0.0064)
    final_cb_params = {
        'max_features': 15000,
        'ngram_range': (1, 2),
        'stop_words': 'english'
    }
    
    # 7. Setup MLflow Tracker
    tracker = MLFlowTracker(experiment_name="3_Production_Models")
    
    print("\n[INFO] Starting Final Content-Based Production Run...")
    with tracker.start_run(run_name="Content_Based_Final_Model"):
        mlflow.log_params(final_cb_params)
        
        print("[INFO] Training final model...")
        model = ContentBasedRecommender(**final_cb_params)
        model.fit_items(meta_df_aligned, text_column='combined_text', item_column='parent_asin')
        model.fit(train_matrix)
        
        print("[INFO] Evaluating final model on validation set...")
        metrics = evaluate_model_at_k(model, val_matrix, eval_users, k=10)
        mlflow.log_metrics(metrics)
        print(f"[INFO] Final Content-Based NDCG@10: {metrics.get('NDCG_10', 0):.4f}")
        
        # Save Model Object locally
        model_path = os.path.join(MODELS_DIR, 'best_cb_model.pkl')
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)
        print(f"[INFO] Model successfully saved locally to: {model_path}")
        
        mlflow.log_artifact(model_path, artifact_path="model_artifact")
        print("[INFO] Model artifact successfully logged to MLflow.")

if __name__ == "__main__":
    run_final_cb_production()