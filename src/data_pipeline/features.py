import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix
from src.data_pipeline.preprocess import _validate_columns

def add_user_segment(df: pd.DataFrame, light_max: int = 2, medium_max: int = 10) -> pd.DataFrame:
    """
    Adds user segment column based on review activity.
    
    Args:
        df: Input dataframe
        light_max: Maximum reviews for light users
        medium_max: Maximum reviews for medium users
    Returns:
        Dataframe with user_segment column added
    """
    _validate_columns(df, ['user_id', 'rating'], required=True)

    # count reviews per user then map back to each row
    user_counts = df.groupby('user_id')['rating'].count()
    df['user_activity'] = df['user_id'].map(user_counts)

    # segment users based on activity thresholds
    df['user_segment'] = pd.cut(df['user_activity'],
                                bins=[0, light_max, medium_max, np.inf],
                                labels=['Light', 'Medium', 'Heavy'])

    # drop temp column used for segmentation
    df.drop(columns=['user_activity'], inplace=True)
    print(f"[INFO] User segments:\n{df['user_segment'].value_counts()}")
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Adds engineered features to the dataframe.
    
    Args:
        df: Input dataframe
    Returns:
        Dataframe with new features added
    """
    _validate_columns(df, ['user_id', 'verified_purchase', 'asin', 'rating', 'timestamp'], required=True)

    # ratio of verified purchases per user — measures user trustworthiness
    df['user_verified_ratio'] = df['user_id'].map(
        df.groupby('user_id')['verified_purchase'].mean()
    )

    # average rating per item — measures overall item quality
    df['item_avg_rating'] = df['asin'].map(
        df.groupby('asin')['rating'].mean()
    )

    # whether the review was made on a weekend — captures different buying behavior
    df['is_weekend'] = (df['timestamp'].dt.dayofweek >= 5).astype(int)

    print(f"[INFO] Added features: user_verified_ratio, item_avg_rating, is_weekend")
    return df


def build_user_item_matrix(df: pd.DataFrame,
                           user_col: str = 'user_id',
                           item_col: str = 'asin',
                           rating_col: str = 'rating') -> tuple:
    """
    Builds a sparse user-item interaction matrix.
    
    Args:
        df: Input dataframe
        user_col: User column name
        item_col: Item column name
        rating_col: Rating column name
    Returns:
        Tuple of (sparse matrix, pivot dataframe)
    """
    _validate_columns(df, [user_col, item_col, rating_col], required=True)

    # create dense pivot table first
    pivot_df = df.pivot_table(index=user_col,
                              columns=item_col,
                              values=rating_col,
                              fill_value=0)

    # convert to sparse matrix to save memory
    matrix = csr_matrix(pivot_df.values)

    print(f"[INFO] Matrix shape: {matrix.shape}")
    print(f"[INFO] Sparsity: {1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1]):.4%}")

    return matrix, pivot_df