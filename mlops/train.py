"""
Model training functions.
"""

import numpy as np
import pandas as pd
import scipy.sparse as sparse
from typing import Tuple

from src.models.cf_model import *


def prepare_sparse_matrices(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame
) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """
    Creates aligned sparse matrices from pre-split Train and Validation DataFrames.
    
    Args:
        train_df: Training DataFrame with user_id and parent_asin
        val_df: Validation DataFrame with user_id and parent_asin
        
    Returns:
        Tuple of (train_matrix, val_matrix, users_to_evaluate)
    """
    unique_users = pd.concat([train_df['user_id'], val_df['user_id']]).unique()
    unique_items = pd.concat([train_df['parent_asin'], val_df['parent_asin']]).unique()
    
    user_to_idx = {user: idx for idx, user in enumerate(unique_users)}
    item_to_idx = {item: idx for idx, item in enumerate(unique_items)}
    
    n_users = len(unique_users)
    n_items = len(unique_items)
    
    train_users = train_df['user_id'].map(user_to_idx).values
    train_items = train_df['parent_asin'].map(item_to_idx).values
    train_matrix = sparse.csr_matrix(
        (np.ones(len(train_df)), (train_users, train_items)),
        shape=(n_users, n_items)
    )
    
    val_users = val_df['user_id'].map(user_to_idx).values
    val_items = val_df['parent_asin'].map(item_to_idx).values
    val_matrix = sparse.csr_matrix(
        (np.ones(len(val_df)), (val_users, val_items)),
        shape=(n_users, n_items)
    )
    
    users_to_evaluate = np.unique(val_users)
    
    return train_matrix, val_matrix, users_to_evaluate


def train_model(model, train_matrix):
    """
    Generic training function for any model implementing BaseRecommender.
    
    Args:
        model: Model instance (ALS, BPR, SVD, ItemKNN)
        train_matrix: Training sparse matrix
        
    Returns:
        Trained model
    """
    model.fit(train_matrix)
    return model


def train_als(
    train_df: pd.DataFrame, 
    val_df: pd.DataFrame, 
    factors: int = 100, 
    iterations: int = 15, 
    regularization: float = 0.01,
    random_state: int = 42
):
    """
    Trains an ALS model (legacy function for backward compatibility).
    
    Use train_model() with ALSRecommender() for new code.
    """
    train_matrix, val_matrix, test_users = prepare_sparse_matrices(train_df, val_df)
    
    model = ALSRecommender(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=random_state
    )
    model.fit(train_matrix)
    
    return model, train_matrix, val_matrix, test_users
