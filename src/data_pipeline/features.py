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
                           total_users: int,
                           total_items: int,
                           user_col: str = 'user_id',
                           item_col: str = 'parent_asin',
                           rating_col: str = 'rating') -> csr_matrix:
    """
    Builds a sparse user-item interaction matrix with fixed dimensions.
    
    Fixing the dimensions ensures that the matrix shape remains consistent 
    across Train, Validation, and Test sets. This is strictly required for 
    matrix multiplication and model evaluation without index mismatch errors.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the interactions.
        total_users (int): Total number of unique users globally (from LabelEncoder).
        total_items (int): Total number of unique items globally (from LabelEncoder).
        user_col (str): Column name for users. Defaults to 'user_id'.
        item_col (str): Column name for items. Defaults to 'asin'.
        rating_col (str): Column name for ratings. Defaults to 'rating'.
        
    Returns:
        csr_matrix: A sparse matrix of shape (total_users, total_items).
    """
    _validate_columns(df, [user_col, item_col, rating_col], required=True)

    # Extract values as numpy arrays for fast matrix construction
    users = df[user_col].values
    items = df[item_col].values
    ratings = df[rating_col].values

    # Build the sparse matrix with the globally fixed shape
    matrix = csr_matrix((ratings, (users, items)), shape=(total_users, total_items))

    # Calculate and log sparsity
    sparsity = 1 - matrix.nnz / (matrix.shape[0] * matrix.shape[1])
    
    print(f"[INFO] Matrix built successfully with shape: {matrix.shape}")
    print(f"[INFO] Matrix sparsity: {sparsity:.4%}")
    
    return matrix


def add_store_tier(df_meta: pd.DataFrame,
                   small_max: int = 2,
                   large_min: int = 50,
                   rating_threshold: float = 4.4) -> pd.DataFrame:
    """
    Adds a combined store_tier column based on product count and average rating.
    
    Args:
        df_meta: Metadata dataframe
        small_max: Max products for small sellers
        large_min: Min products for large/premium stores
        rating_threshold: Min avg rating for premium classification
    Returns:
        Dataframe with store_tier column added (Premium / Trusted / Small Seller)
    """
    _validate_columns(df_meta, ['store', 'parent_asin', 'average_rating'], required=True)

    store_stats = df_meta.groupby('store').agg(
        product_count=('parent_asin', 'count'),
        avg_rating=('average_rating', 'mean')
    ).reset_index()

    def classify(row):
        if row['product_count'] >= large_min and row['avg_rating'] >= rating_threshold:
            return 'Premium'
        elif row['product_count'] > small_max or row['avg_rating'] >= 3.6:
            return 'Trusted'
        else:
            return 'Small Seller'

    store_stats['store_tier'] = store_stats.apply(classify, axis=1)

    df_meta = df_meta.merge(
        store_stats[['store', 'store_tier']],
        on='store',
        how='left'
    )

    print(f"[INFO] Store tier distribution:")
    print(df_meta['store_tier'].value_counts())

    return df_meta