"""
Compare multiple CF models on the same dataset.
"""
import os
import sys
import time
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Dict, Any, List
sys.path.append('..')
os.chdir('..')

from src.models.cf_model import *
from mlops.train import *
from mlops.evaluate import *
from mlops.mlflow_tracking import *
from src.models.cf_model import get_model_params


def get_model(model_name: str, params: Dict[str, Any]):
    """
    Factory function to create model instance.
    
    Args:
        model_name: Name of the model
        params: Model parameters
        
    Returns:
        Instantiated model
    """
    if model_name == "ALS":
        return ALSRecommender(**params)
    elif model_name == "BPR":
        return BPRRecommender(**params)
    elif model_name == "SVD":
        return SVDRecommender(**params)
    elif model_name == "ItemKNN":
        return ItemBasedKNN(**params)
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_sample_users(test_users: np.ndarray, n_sample: int, random_state: int = 42) -> np.ndarray:
    """
    Sample users for evaluation (speed optimization).
    
    Args:
        test_users: Array of all test users
        n_sample: Number of users to sample (0 = all)
        random_state: Random seed
        
    Returns:
        Sampled users array
    """
    if n_sample <= 0 or n_sample >= len(test_users):
        return test_users
    
    np.random.seed(random_state)
    return np.random.choice(test_users, size=n_sample, replace=False)


def train_and_evaluate(
    model_name: str,
    model_params: Dict[str, Any],
    train_matrix,
    val_matrix,
    sample_users: np.ndarray,
    k: int,
    tracker: MLFlowTracker
) -> Dict[str, Any]:
    """
    Train and evaluate a single model.
    
    Args:
        model_name: Name of the model
        model_params: Model hyperparameters
        train_matrix: Training sparse matrix
        val_matrix: Validation sparse matrix
        sample_users: Users to evaluate on
        k: Top-K for evaluation
        tracker: MLflow tracker
        
    Returns:
        Dictionary of metrics
    """
    print(f"\n{'='*60}")
    print(f" Training: {model_name}")
    print(f"{'='*60}")
    
    try:
        with tracker.start_run(run_name=f"{model_name}_comparison"):
            
            # Create model
            model = get_model(model_name, model_params)
            
            # Train
            train_start = time.time()
            model = train_model(model, train_matrix)
            train_time = time.time() - train_start
            
            # Evaluate
            eval_start = time.time()
            metrics = evaluate_model_at_k(
                model=model,
                val_matrix=val_matrix,
                users_to_evaluate=sample_users,
                k=k
            )
            eval_time = time.time() - eval_start
            
            # Add timing
            metrics['train_time_sec'] = round(train_time, 2)
            metrics['eval_time_sec'] = round(eval_time, 2)
            
            # Log to MLflow
            tracker.log_params({
                'model_type': model_name,
                'n_train_users': train_matrix.shape[0],
                'n_train_items': train_matrix.shape[1],
                'n_val_users': len(sample_users),
                'k': k,
                **model_params
            })
            tracker.log_metrics(metrics)
            
            # Print results
            print(f" {model_name} trained in {train_time:.1f}s | evaluated in {eval_time:.1f}s")
            for metric, value in metrics.items():
                if not metric.endswith('_sec'):
                    print(f"   {metric}: {value}")
            
            return metrics
    
    except Exception as e:
        print(f" {model_name} failed: {e}")
        return {'error': str(e)}


def compare_all_models(
    config: Dict[str, Any],
    models_to_compare: List[str] = None
) -> pd.DataFrame:
    """
    Compare all configured CF models.
    
    Args:
        config: Configuration dictionary
        models_to_compare: List of model names to compare (None = all)
        
    Returns:
        DataFrame with comparison results
    """
    # Load data
    paths = config['experiment']
    train_df = pd.read_parquet(paths['train_path'])
    val_df = pd.read_parquet(paths['val_path'])
    
    print(f" Train: {train_df.shape}")
    print(f" Val:   {val_df.shape}")
    
    # Prepare sparse matrices
    print("\n Preparing sparse matrices...")
    start = time.time()
    train_matrix, val_matrix, test_users = prepare_sparse_matrices(train_df, val_df)
    print(f"   Done in {time.time() - start:.1f}s")
    print(f"   Train: {train_matrix.shape} | nnz: {train_matrix.nnz:,}")
    print(f"   Val:   {val_matrix.shape} | nnz: {val_matrix.nnz:,}")
    
    # Setup MLflow
    tracker = MLFlowTracker(
        experiment_name=paths['name'],
        db_uri=paths['mlflow_uri']
    )
    
    # Get sample users
    eval_cfg = config['evaluation']
    sample_users = get_sample_users(
        test_users,
        n_sample=eval_cfg['n_sample_users'],
        random_state=eval_cfg['random_state']
    )
    print(f"\n Evaluating on {len(sample_users):,} users (out of {len(test_users):,})")
    
    # Models to compare
    if models_to_compare is None:
        models_to_compare = ['ALS', 'BPR', 'SVD', 'ItemKNN']
    
    # Train and evaluate each
    results = {}
    for model_name in models_to_compare:
        model_params = get_model_params(config, model_name)
        metrics = train_and_evaluate(
            model_name=model_name,
            model_params=model_params,
            train_matrix=train_matrix,
            val_matrix=val_matrix,
            sample_users=sample_users,
            k=eval_cfg['k'],
            tracker=tracker
        )
        results[model_name] = metrics
    
    # Create comparison DataFrame
    comparison_df = pd.DataFrame(results).T
    comparison_df = comparison_df.sort_values(f"HitRate_{eval_cfg['k']}", ascending=False)
    
    return comparison_df


def plot_comparison(
    comparison_df: pd.DataFrame,
    output_dir: Path,
    k: int = 10
):
    """
    Plot model comparison metrics.
    
    Args:
        comparison_df: DataFrame with comparison results
        output_dir: Directory to save plots
        k: K value used in evaluation
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Metrics to plot
    metrics_to_plot = [
        f'HitRate_{k}', f'Precision_{k}', f'Recall_{k}',
        f'MRR_{k}', f'NDCG_{k}', f'Adj_Precision_{k}'
    ]
    
    # Filter available metrics
    metrics_to_plot = [m for m in metrics_to_plot if m in comparison_df.columns]
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 10))
    
    for idx, metric in enumerate(metrics_to_plot):
        ax = axes[idx // 3, idx % 3]
        
        values = comparison_df[metric].astype(float)
        bars = ax.barh(comparison_df.index, values, color='steelblue', edgecolor='black')
        ax.set_xlabel(metric, fontsize=11)
        ax.set_title(f'{metric} Comparison', fontsize=12, fontweight='bold')
        ax.invert_yaxis()
        
        for bar, val in zip(bars, values):
            ax.text(val + 0.001, bar.get_y() + bar.get_height()/2,
                    f'{val:.4f}', va='center', fontsize=9)
    
    # Remove empty subplot
    if len(metrics_to_plot) < 6:
        axes[1, 2].axis('off')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'model_comparison.png'}")


def plot_time_comparison(comparison_df: pd.DataFrame, output_dir: Path):
    """Plot training/evaluation time comparison."""
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    if 'train_time_sec' not in comparison_df.columns:
        return
    
    time_df = comparison_df[['train_time_sec', 'eval_time_sec']].astype(float)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    x = range(len(time_df))
    width = 0.35
    ax.bar([i - width/2 for i in x], time_df['train_time_sec'], width, label='Train', color='steelblue')
    ax.bar([i + width/2 for i in x], time_df['eval_time_sec'], width, label='Eval', color='coral')
    ax.set_xticks(x)
    ax.set_xticklabels(time_df.index)
    ax.set_ylabel('Time (seconds)', fontsize=12)
    ax.set_title('Training & Evaluation Time per Model', fontsize=14, fontweight='bold')
    ax.legend()
    ax.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(output_dir / 'model_time_comparison.png', dpi=100, bbox_inches='tight')
    plt.close()
    print(f"Saved: {output_dir / 'model_time_comparison.png'}")
