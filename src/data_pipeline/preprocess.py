import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler

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


def replace_none_strings(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Replaces 'None' strings and empty strings with actual NaN values.
    
    Args:
        df: Input dataframe
        columns: List of columns to clean
    Returns:
        Cleaned dataframe
    """
    columns = _validate_columns(df, columns, required=False)
    
    for col in columns:
        df[col] = df[col].replace({'None': None, '': None})
        
    print(f"[INFO] Replaced None strings in: {columns}")
    return df


def remove_missing_values(df: pd.DataFrame, subset: list = None) -> pd.DataFrame:
    """
    Removes rows with missing values (Optimized for Parquet).
    """
    before = len(df)
    if subset:
        subset = _validate_columns(df, subset)
        
        for col in subset:
            if df[col].dtype == 'object':
                df[col] = df[col].replace(r'^\s*$', np.nan, regex=True)
                
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


def handle_outliers(df: pd.DataFrame, columns: list) -> pd.DataFrame:
    """
    Handles outliers by applying Log Transformation (log1p) 
    to preserve the Long-Tail distribution signal without clipping data.
    
    Args:
        df: Input dataframe
        columns: List of column names to transform
    Returns:
        Dataframe with log-transformed columns
    """
    columns = _validate_columns(df, columns, required=True)

    for column in columns:
        df[column] = np.log1p(df[column])
        print(f"[INFO] Applied Log Transformation (log1p) to '{column}'")

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

    valid_texts = (word_count >= min_words).sum()
    print(f"[INFO] Valid texts for NLP: {valid_texts}")
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


def time_based_split(df: pd.DataFrame, column: str = "timestamp", val_year: int = 2021, test_year: int = 2022) -> tuple:
    """
    Splits dataframe into train, validation, and test sets based on time.
    
    This time-based splitting prevents data leakage by ensuring that the 
    model is trained on historical data and evaluated on future data.
    
    Args:
        df (pd.DataFrame): Input dataframe containing the interaction data.
        column (str): The name of the datetime column used for splitting. Defaults to "timestamp".
        val_year (int): The starting year for the validation set. Defaults to 2021.
        test_year (int): The starting year for the test set. Defaults to 2022.
        
    Returns:
        tuple: A tuple containing three pandas DataFrames:
            - train_df: Data before val_year.
            - val_df: Data from val_year up to (but not including) test_year.
            - test_df: Data from test_year onwards.
    """
    from src.data_pipeline.preprocess import _validate_columns
    _validate_columns(df, [column], required=True)
    
    train_df = df[df[column].dt.year < val_year].copy()
    
    val_df   = df[(df[column].dt.year >= val_year) & (df[column].dt.year < test_year)].copy()
    
    test_df  = df[df[column].dt.year >= test_year].copy()

    total = len(df)
    print(f"[INFO] Train: {len(train_df)} rows ({len(train_df)/total*100:.1f}%)")
    print(f"[INFO] Val:   {len(val_df)} rows ({len(val_df)/total*100:.1f}%)")
    print(f"[INFO] Test:  {len(test_df)} rows ({len(test_df)/total*100:.1f}%)")

    return train_df, val_df, test_df


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
