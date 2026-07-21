"""
Hybrid System Two-Phase Evaluation Pipeline (10% Dynamic Tuning Sample).

Phase 1: Rapid Grid Search over alpha candidates using a 10% sample of users.
Phase 2: Comprehensive Full Evaluation over the entire dataset using the winning alpha.
"""

import os
import sys
import torch
import pandas as pd
import yaml
import mlflow
import random

# Resolve project root for imports
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if PROJECT_ROOT not in sys.path:
    sys.path.append(PROJECT_ROOT)

from src.models.ncf_model import NeuralCFRecommender
from src.models.cb_model import ContentBasedRecommender
from src.models.hybrid_model import HybridRecommender
from mlops.train import prepare_sparse_matrices
from mlops.evaluate import evaluate_model_at_k
from mlops.mlflow_tracking import MLFlowTracker


def run_hybrid_two_phase_pipeline():
    # 1. Load Data Configurations
    config_path = os.path.join(PROJECT_ROOT, 'configs', 'data_config.yaml')
    with open(config_path, 'r', encoding='utf-8') as f:
        data_config = yaml.safe_load(f)

    data_dir = os.path.join(PROJECT_ROOT, data_config['paths']['processed_data'])
    models_dir = os.path.join(PROJECT_ROOT, 'models')

    # 2. Load Datasets & Matrices
    print("[INFO] Loading datasets for Hybrid Two-Phase Pipeline...")
    train_df = pd.read_parquet(os.path.join(data_dir, data_config['paths']['train_file']))
    val_df = pd.read_parquet(os.path.join(data_dir, data_config['paths']['val_file']))

    train_df = train_df[train_df['timestamp'] >= '2017-01-01']
    train_matrix, val_matrix, full_eval_users = prepare_sparse_matrices(train_df, val_df)
    num_users, num_items = train_matrix.shape

    # Dynamic 10% Sampling for Phase 1 Tuning
    random.seed(42)
    sample_rate = 0.10
    n_sample = int(len(full_eval_users) * sample_rate)
    sample_eval_users = random.sample(list(full_eval_users), n_sample)
    print(f"[INFO] Total validation users: {len(full_eval_users)}")
    print(f"[INFO] 10% Sample size calculated for Phase 1 Tuning: {n_sample} users.")

    # 3. Initialize & Load Pre-trained NCF Model
    print("[INFO] Loading Neural CF weights from best_ncf_model.pt...")
    ncf_engine = NeuralCFRecommender(num_users=num_users, num_items=num_items, emb_dim=32)
    ncf_weights_path = os.path.join(models_dir, 'best_ncf_model.pt')
    ncf_engine.model.load_state_dict(torch.load(ncf_weights_path, weights_only=True))
    ncf_engine.fit(train_matrix)

    # 4. Initialize Content-Based Engine and Handle Metadata
    print("[INFO] Initializing Content-Based engine...")
    cb_engine = ContentBasedRecommender()

    metadata_filename = data_config['paths']['metadata_file']
    metadata_path = os.path.join(data_dir, metadata_filename)
    items_df = pd.read_parquet(metadata_path)

    if 'combined_text' not in items_df.columns:
        text_cols = [col for col in ['title', 'text', 'description', 'features', 'categories'] if col in items_df.columns]
        items_df['combined_text'] = items_df[text_cols].fillna("").astype(str).agg(' '.join, axis=1)

    cb_engine.fit_items(items_df, text_column='combined_text', item_column='parent_asin')
    cb_engine.fit(train_matrix)

    # 5. Define Grid Search Space
    alpha_candidates = [0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95]

    # 6. Initialize MLflow Master Tracker
    tracker = MLFlowTracker(experiment_name="3_Production_Models")

    print("\n" + "="*60)
    print("LAUNCHING HYBRID TWO-PHASE PIPELINE (10% DYNAMIC SAMPLE)")
    print("="*60)

    with tracker.start_run(run_name="Hybrid_Two_Phase_10Percent_Master"):
        
        # ------------------------------------------------------------
        # PHASE 1: TUNING VIA 10% SAMPLE
        # ------------------------------------------------------------
        print(f"\n[PHASE 1] Starting Grid Search on 10% Sample Pool ({n_sample} users)...")
        best_alpha = None
        best_sample_ndcg = -1.0

        for alpha in alpha_candidates:
            trial_name = f"Grid_Search_10Percent_Alpha_{alpha}"
            with mlflow.start_run(run_name=trial_name, nested=True):
                hybrid_system = HybridRecommender(ncf_model=ncf_engine, cb_model=cb_engine, alpha=alpha)
                hybrid_system.fit(train_matrix)

                metrics = evaluate_model_at_k(hybrid_system, val_matrix, sample_eval_users, k=10)
                
                mlflow.log_param("alpha_candidate", alpha)
                mlflow.log_param("evaluation_type", "10Percent_Sample")
                mlflow.log_metrics({f"sample_{k}": v for k, v in metrics.items()})

                current_ndcg = metrics.get('NDCG_10', 0.0)
                print(f" -> Candidate Alpha {alpha} | Sample HitRate@10: {metrics.get('HitRate_10', 0):.4f} | Sample NDCG@10: {current_ndcg:.4f}")

                if current_ndcg > best_sample_ndcg:
                    best_sample_ndcg = current_ndcg
                    best_alpha = alpha

        print("\n" + "="*60)
        print(f"[PHASE 1 COMPLETE] Winner Alpha found on 10% Sample: {best_alpha}")
        print("="*60)

        # ------------------------------------------------------------
        # PHASE 2: FULL PRODUCTION EVALUATION
        # ------------------------------------------------------------
        print(f"\n[PHASE 2] Executing Full Production Evaluation using Winner Alpha = {best_alpha}...")
        print(f"[INFO] Computing performance over all {len(full_eval_users)} validation users. Please hold...")

        with mlflow.start_run(run_name=f"Full_Production_Alpha_{best_alpha}", nested=True):
            final_hybrid_system = HybridRecommender(ncf_model=ncf_engine, cb_model=cb_engine, alpha=best_alpha)
            final_hybrid_system.fit(train_matrix)

            final_metrics = evaluate_model_at_k(final_hybrid_system, val_matrix, full_eval_users, k=10)

            # Log production metrics
            mlflow.log_param("optimal_best_alpha", best_alpha)
            mlflow.log_param("evaluation_type", "Full Dataset")
            mlflow.log_metrics(final_metrics)

            print("\n" + "="*60)
            print("FINAL FULL PIPELINE RESULTS")
            print("="*60)
            print(f"Optimal Alpha Selected: {best_alpha}")
            print(f"Final HitRate@10:       {final_metrics.get('HitRate_10', 0.0):.4f}")
            print(f"Final NDCG@10:          {final_metrics.get('NDCG_10', 0.0):.4f}")
            print("="*60)
            
        # Log summary parameters to the root master run
        mlflow.log_param("champion_alpha", best_alpha)
        mlflow.log_metrics({f"champion_full_{k}": v for k, v in final_metrics.items()})
        print("[SUCCESS] Tuning trials and full production metrics securely locked in MLflow.")


if __name__ == "__main__":
    run_hybrid_two_phase_pipeline()