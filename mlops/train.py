import numpy as np
import pandas as pd
import scipy.sparse as sparse
from implicit.als import AlternatingLeastSquares
from typing import Tuple

def prepare_sparse_matrices(train_df: pd.DataFrame, val_df: pd.DataFrame) -> Tuple[sparse.csr_matrix, sparse.csr_matrix, np.ndarray]:
    """
    Creates aligned sparse matrices from pre-split Train and Validation DataFrames.
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

def train_als(train_df: pd.DataFrame, val_df: pd.DataFrame, factors: int = 100, iterations: int = 15, regularization: float = 0.01):
    """
    Trains the Implicit ALS model and returns necessary matrices.
    """
    train_matrix, val_matrix, test_users = prepare_sparse_matrices(train_df, val_df)

    model = AlternatingLeastSquares(
        factors=factors,
        iterations=iterations,
        regularization=regularization,
        random_state=42,
        calculate_training_loss=False
    )

    model.fit(train_matrix)

    # Added train_matrix to the returned variables
    return model, train_matrix, val_matrix, test_users