import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from scipy.sparse import csr_matrix


def _validate_columns(df: pd.DataFrame, columns: list, required: bool = False) -> list:
    """
    Validates that columns exist in dataframe.
    
    Args:
        df: Input dataframe
        columns: List of columns to validate
        required: If True, raises error if any column is missing
    Returns:
        List of valid columns
    """
    valid = [c for c in columns if c in df.columns]
    invalid = [c for c in columns if c not in df.columns]
    
    if invalid and required:
        raise ValueError(f"Required columns missing: {invalid}")
    elif invalid:
        print(f"[WARNING] Columns not found, skipping: {invalid}")
    
    return valid


def remove_missing_values(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Removes rows with missing values.
    
    Args:
        df: Input dataframe
        subset: List of columns to check for nulls (None = all columns)
    Returns:
        Cleaned dataframe
    """
    before = len(df)
    if subset:
        subset = _validate_columns(df, subset)
    df.dropna(subset=subset if subset else None, inplace=True)
    after = len(df)
    print(f"[INFO] Removed {before - after} rows with missing values")
    return df


def remove_duplicates(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Removes duplicate rows from the dataframe.
    
    Args:
        df: Input dataframe
        subset: List of columns to consider for duplicates (None = all columns)
    Returns:
        Cleaned dataframe
    """
    if subset:
        subset = _validate_columns(df, subset)
    before = len(df)
    df.drop_duplicates(subset=subset, inplace=True)
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows")
    return df


def drop_useless_columns(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Drops specified columns from the dataframe.
    
    Args:
        df: Input dataframe
        columns: List of column names to drop
    Returns:
        Cleaned dataframe
    """
    cols_to_drop = _validate_columns(df, columns)
    df.drop(cols_to_drop, axis=1, inplace=True)
    print(f"[INFO] Dropped columns: {cols_to_drop}")
    return df


def handle_outliers(df: pd.DataFrame, columns: list, method: str = "upper") -> pd.DataFrame:
    """
    Caps outliers in specified columns using IQR method.
    
    Args:
        df: Input dataframe
        columns: List of column names to handle outliers
        method: "upper", "lower", or "both"
    Returns:
        Dataframe with capped outliers
    """
    method = method.lower()
    if method not in ["upper", "lower", "both"]:
        raise ValueError(f"method must be 'upper', 'lower', or 'both'. Got: {method}")

    columns = _validate_columns(df, columns, required=True)

    for column in columns:
        Q1 = df[column].quantile(0.25)
        Q3 = df[column].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR

        if method == "upper":
            outliers = (df[column] > upper).sum()
            df[column] = df[column].clip(upper=upper)
        elif method == "lower":
            outliers = (df[column] < lower).sum()
            df[column] = df[column].clip(lower=lower)
        elif method == "both":
            outliers = ((df[column] > upper) | (df[column] < lower)).sum()
            df[column] = df[column].clip(lower=lower, upper=upper)

        print(f"[INFO] Capped {outliers} outliers in '{column}' using method='{method}'")

    return df


def convert_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """
    Converts timestamp column to datetime format.
    
    Args:
        df: Input dataframe
        column: Timestamp column name
    Returns:
        Dataframe with converted timestamp
    """
    _validate_columns(df, [column], required=True)
    df[column] = pd.to_datetime(df[column], unit='ms', errors='coerce')
    print(f"[INFO] Converted '{column}' to datetime")
    return df


def convert_to_integer(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Converts specified columns to integer type.
    
    Args:
        df: Input dataframe
        columns: List of column names to convert
    Returns:
        Dataframe with converted columns
    """
    columns = _validate_columns(df, columns, required=True)
    for column in columns:
        df[column] = df[column].astype(int)
        print(f"[INFO] Converted '{column}' to integer")
    return df


def detect_spam(df: pd.DataFrame, max_reviews_per_day: int = 5, min_time_gap: int = 10) -> pd.DataFrame:
    """
    Removes spam users based on review frequency and time gaps.
    
    Args:
        df: Input dataframe
        max_reviews_per_day: Maximum allowed reviews per day
        min_time_gap: Minimum seconds between reviews
    Returns:
        Dataframe with spam users removed
    """
    _validate_columns(df, ['user_id', 'timestamp', 'rating'], required=True)

    before = len(df['user_id'].unique())

    # Step 1: calculate reviews per day per user
    df['date'] = df['timestamp'].dt.date
    df['reviews_per_day'] = df.groupby(['user_id', 'date'])['rating'].transform('count')

    # Step 2: calculate time gap in seconds between consecutive reviews
    df = df.sort_values(['user_id', 'timestamp'])
    df['time_diff'] = df.groupby('user_id')['timestamp'].diff().dt.total_seconds()

    # Step 3: identify spam users by either condition
    spam_users = df[
        (df['reviews_per_day'] > max_reviews_per_day) |
        (df['time_diff'] < min_time_gap)
    ]['user_id'].unique()

    # Step 4: remove all reviews from spam users
    df = df[~df['user_id'].isin(spam_users)]

    # Step 5: drop temporary columns used for detection
    df.drop(columns=['date', 'reviews_per_day', 'time_diff'], inplace=True)

    after = len(df['user_id'].unique())
    print(f"[INFO] Removed {before - after} spam users")
    return df


def add_review_weight(df: pd.DataFrame, verified_weight: float = 1.0, unverified_weight: float = 0.7) -> pd.DataFrame:
    """
    Adds a weight column based on verified purchase status.
    
    Args:
        df: Input dataframe
        verified_weight: Weight for verified purchases
        unverified_weight: Weight for unverified purchases
    Returns:
        Dataframe with weight column added
    """
    _validate_columns(df, ['verified_purchase'], required=True)
    df['weight'] = df['verified_purchase'].map({1: verified_weight, 0: unverified_weight})
    print(f"[INFO] Added weight column: verified={verified_weight}, unverified={unverified_weight}")
    return df


def filter_text(df: pd.DataFrame, column: str = "text", min_words: int = 5, max_words: int = 250) -> pd.DataFrame:
    """
    Filters text column based on word count.
    Short texts are marked as None, long texts are truncated.
    
    Args:
        df: Input dataframe
        column: Text column name
        min_words: Minimum number of words
        max_words: Maximum number of words
    Returns:
        Dataframe with filtered text
    """
    _validate_columns(df, [column], required=True)

    # count words per review
    word_count = df[column].str.split().str.len()
    short = (word_count < min_words).sum()
    long  = (word_count > max_words).sum()

    # mark short reviews as None — will use rating only
    df.loc[word_count < min_words, column] = None

    # truncate long reviews to max_words
    df.loc[word_count > max_words, column] = df.loc[word_count > max_words, column]\
        .str.split().str[:max_words].str.join(' ')

    print(f"[INFO] Marked {short} short texts as None")
    print(f"[INFO] Truncated {long} long texts to {max_words} words")
    return df


def encode_labels(df: pd.DataFrame, columns: list, encoders: dict = None) -> tuple:
    """
    Encodes categorical columns to integer labels.
    
    Args:
        df: Input dataframe
        columns: List of column names to encode
        encoders: Dict of fitted encoders (None = fit new encoders on train)
    Returns:
        Tuple of (dataframe, encoders dict)
    """
    columns = _validate_columns(df, columns, required=True)

    # initialize encoders dict if not provided (train mode)
    if encoders is None:
        encoders = {}

    for column in columns:
        if column not in encoders:
            # fit new encoder on train data
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column])
        else:
            # use existing encoder on test data
            df[column] = encoders[column].transform(df[column])
        print(f"[INFO] Encoded '{column}' — {df[column].nunique()} unique labels")

    return df, encoders


def time_based_split(df: pd.DataFrame, column: str = "timestamp", test_year: int = 2022) -> tuple:
    """
    Splits dataframe into train and test sets based on time.
    
    Args:
        df: Input dataframe
        column: Timestamp column name
        test_year: Year to split on (test = this year onwards)
    Returns:
        Tuple of (train_df, test_df)
    """
    _validate_columns(df, [column], required=True)

    train_df = df[df[column].dt.year < test_year]
    test_df  = df[df[column].dt.year >= test_year]

    train_ratio = len(train_df) / len(df) * 100
    test_ratio  = len(test_df) / len(df) * 100

    print(f"[INFO] Train: {len(train_df)} rows ({train_ratio:.1f}%)")
    print(f"[INFO] Test:  {len(test_df)} rows ({test_ratio:.1f}%)")

    return train_df, test_df


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


def normalize(df: pd.DataFrame, column: str, scaler=None) -> tuple:
    """
    Normalizes a single column using MinMaxScaler.
    
    Args:
        df: Input dataframe
        column: Column name to normalize
        scaler: Fitted scaler (None = fit new scaler on train)
    Returns:
        Tuple of (dataframe, scaler)
    """
    _validate_columns(df, [column], required=True)

    if scaler is None:
        # fit and transform on train data
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[[column]])
        print(f"[INFO] Fitted and normalized '{column}'")
    else:
        # transform only on test data using train scaler
        df[column] = scaler.transform(df[[column]])
        print(f"[INFO] Transformed '{column}' using existing scaler")

    return df, scaler


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