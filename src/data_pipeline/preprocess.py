"""Data Preprocessing Module for Recommendation System.

This module provides comprehensive data preprocessing functions organized
in logical pipeline stages: cleaning -> filtering -> encoding -> splitting.

Pipeline Order:
    1. Data Cleaning (remove missing, duplicates, outliers)
    2. Text Filtering & Weight Assignment
    3. Spam Detection
    4. High-Signal Rating Filtering (rating >= 4.0)
    5. User-Item Deduplication
    6. Iterative K-Core Filtering (users + items)
    7. Top-N Item Reduction
    8. Confidence Weight Assignment
    9. Label Encoding
    10. Time-Based Splitting
    11. Normalization

"""

from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, MinMaxScaler


def _validate_columns(
    df: pd.DataFrame,
    columns: List[str],
    required: bool = False
) -> List[str]:
    """Validate that specified columns exist in dataframe.

    Helper function to check column existence and provide consistent
    feedback. Used internally by all functions that access DataFrame
    columns to ensure robust error handling.

    Args:
        df (pd.DataFrame): Input dataframe to validate.
        columns (List[str]): List of column names to check for existence.
        required (bool, optional): If True, raises ValueError when columns
            are missing. If False, issues a warning only. Defaults to False.

    Returns:
        List[str]: List of valid columns that exist in the dataframe.

    Raises:
        ValueError: If required=True and any specified columns are missing
            from the dataframe.
    """
    valid = [c for c in columns if c in df.columns]
    invalid = [c for c in columns if c not in df.columns]

    if invalid and required:
        raise ValueError(f"Required columns missing: {invalid}")
    elif invalid:
        print(f"[WARNING] Columns not found, skipping: {invalid}")

    return valid


def replace_none_strings(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Replace 'None' string representations with proper NaN values.

    Converts string representations of missing values ('None', 'none', empty
    strings) to pandas NaN for consistent missing value handling throughout
    the preprocessing pipeline.

    Args:
        df (pd.DataFrame): Input dataframe containing columns to clean.
        columns (List[str]): List of column names where None strings should
            be replaced. Non-existent columns are skipped with a warning.

    Returns:
        pd.DataFrame: Dataframe with None string representations replaced
            by proper NaN values.
    """
    columns = _validate_columns(df, columns, required=False)
    for col in columns:
        df[col] = df[col].replace({"None": None, "": None})
    print(f"[INFO] Replaced None strings in: {columns}")
    return df


def remove_missing_values(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove rows with missing (NaN) values in specified columns.

    Handles both explicit NaN values and whitespace-only strings in object-type
    columns. Optimized for Parquet format data. Rows with any missing values in
    the specified subset are completely removed.

    Args:
        df (pd.DataFrame): Input dataframe with potential missing values.
        subset (Optional[List[str]], optional): List of column names to check
            for missing values. If None, checks all columns for missing values.
            Non-existent columns are skipped. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with all rows containing NaN in the specified
            columns removed. Number of removed rows is logged.
    """
    before = len(df)
    if subset:
        subset = _validate_columns(df, subset, required=False)
        for col in subset:
            if df[col].dtype == "object":
                df[col] = df[col].replace(r"^\s*$", np.nan, regex=True)
    df = df.dropna(subset=subset if subset else None)
    after = len(df)
    print(f"[INFO] Removed {before - after} rows with missing values | Remaining: {after}")
    return df


def remove_duplicates(df: pd.DataFrame, subset: Optional[List[str]] = None) -> pd.DataFrame:
    """Remove exact duplicate rows from the dataframe.

    Removes rows that are identical across all columns (or specified subset).
    For user-item pair deduplication in implicit feedback scenarios, use
    deduplicate_user_item() instead.

    Args:
        df (pd.DataFrame): Input dataframe potentially containing duplicates.
        subset (Optional[List[str]], optional): List of column names to consider
            when detecting duplicates. If None, all columns are used to detect
            duplicate rows. Defaults to None.

    Returns:
        pd.DataFrame: Dataframe with duplicate rows removed. Only the first
            occurrence of each duplicate is retained.
    """
    if subset:
        subset = _validate_columns(df, subset, required=False)
    before = len(df)
    df = df.drop_duplicates(subset=subset)
    after = len(df)
    print(f"[INFO] Removed {before - after} duplicate rows | Remaining: {after}")
    return df


def drop_useless_columns(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Remove unnecessary columns from the dataframe.

    Drops specified columns that are not needed for downstream processing.
    Validates column existence before attempting removal. Non-existent columns
    are skipped silently.

    Args:
        df (pd.DataFrame): Input dataframe containing columns to drop.
        columns (List[str]): List of column names to remove from the dataframe.
            Non-existent columns trigger a warning but do not cause an error.

    Returns:
        pd.DataFrame: Dataframe with specified columns removed. Remaining
            columns retain their original order.
    """
    cols_to_drop = _validate_columns(df, columns, required=False)
    df = df.drop(cols_to_drop, axis=1)
    print(f"[INFO] Dropped columns: {cols_to_drop}")
    return df


def handle_outliers(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Handle outliers using log transformation (log1p).

    Applies log1p transformation to specified columns to handle outliers while
    preserving long-tail distribution signals. This approach is effective for
    columns with exponential or power-law distributions without clipping or
    removing data.

    Args:
        df (pd.DataFrame): Input dataframe with numeric columns potentially
            containing outliers.
        columns (List[str]): List of numeric column names to transform.
            All specified columns must exist; missing columns raise ValueError.

    Returns:
        pd.DataFrame: Dataframe with log1p-transformed columns. Original
            column values are replaced by transformed values.

    Raises:
        ValueError: If any specified column does not exist in the dataframe.
    """
    columns = _validate_columns(df, columns, required=True)
    for column in columns:
        df[column] = np.log1p(df[column])
        print(f"[INFO] Applied log1p transformation to '{column}'")
    return df


def convert_timestamp(df: pd.DataFrame, column: str = "timestamp") -> pd.DataFrame:
    """Convert Unix millisecond timestamp column to pandas datetime format.

    Converts Unix millisecond timestamps to pandas datetime objects for
    time-based operations, analysis, and filtering.

    Args:
        df (pd.DataFrame): Input dataframe with timestamp column to convert.
        column (str, optional): Name of the timestamp column to convert.
            Must be in Unix milliseconds format. Defaults to "timestamp".

    Returns:
        pd.DataFrame: Dataframe with the specified column converted to
            datetime64[ns] format. Invalid timestamps are converted to NaT.

    Raises:
        ValueError: If the specified column does not exist in the dataframe.
    """
    _validate_columns(df, [column], required=True)
    df[column] = pd.to_datetime(df[column], unit="ms", errors="coerce")
    print(f"[INFO] Converted '{column}' to datetime format")
    return df


def convert_to_integer(df: pd.DataFrame, columns: List[str]) -> pd.DataFrame:
    """Convert specified columns to integer data type.

    Ensures numeric columns are represented as integers for memory efficiency
    and downstream processing. Useful for categorical IDs, counts, and ratings.

    Args:
        df (pd.DataFrame): Input dataframe with columns to convert.
        columns (List[str]): List of column names to convert to integer type.
            All specified columns must exist; missing columns raise ValueError.

    Returns:
        pd.DataFrame: Dataframe with specified columns converted to int64 type.
            Column values are replaced with integer representations.

    Raises:
        ValueError: If any specified column does not exist in the dataframe.
    """
    columns = _validate_columns(df, columns, required=True)
    for column in columns:
        df[column] = df[column].astype(int)
        print(f"[INFO] Converted '{column}' to integer type")
    return df


def detect_spam(df: pd.DataFrame, max_reviews_per_day: int = 5, min_time_gap: int = 10) -> pd.DataFrame:
    """Detect and remove spam users based on abnormal review patterns.

    Identifies suspicious users exhibiting anomalous behavior by tracking:
    - Users posting more than max_reviews_per_day on the same day
    - Users posting with less than min_time_gap seconds between consecutive
    
    Removes ALL reviews from flagged spam users to eliminate contaminated data.

    Args:
        df (pd.DataFrame): Input dataframe with user interactions containing
            'user_id', 'timestamp', and 'rating' columns.
        max_reviews_per_day (int, optional): Maximum number of reviews allowed
            per user per calendar day. Exceeding this triggers spam flag.
            Defaults to 5.
        min_time_gap (int, optional): Minimum seconds required between
            consecutive reviews by the same user. Reviews violating this
            trigger spam flag. Defaults to 10.

    Returns:
        pd.DataFrame: Dataframe with all reviews from spam users removed.
            Temporary detection columns are dropped. Logs show user and row
            count changes.

    Raises:
        ValueError: If required columns ('user_id', 'timestamp', 'rating')
            do not exist in the dataframe.
    """
    _validate_columns(df, ["user_id", "timestamp", "rating"], required=True)
    before_users = df["user_id"].nunique()
    before_rows = len(df)
    
    df["_date"] = df["timestamp"].dt.date
    df["_reviews_per_day"] = df.groupby(["user_id", "_date"])["rating"].transform("count")
    df = df.sort_values(["user_id", "timestamp"])
    df["_time_diff"] = df.groupby("user_id")["timestamp"].diff().dt.total_seconds()
    
    spam_users = df[(df["_reviews_per_day"] > max_reviews_per_day) | (df["_time_diff"] < min_time_gap)]["user_id"].unique()
    df = df[~df["user_id"].isin(spam_users)]
    df = df.drop(columns=["_date", "_reviews_per_day", "_time_diff"])
    
    after_users = df["user_id"].nunique()
    after_rows = len(df)
    print(f"[INFO] Spam detection: Removed {before_users - after_users} spam users")
    print(f"[INFO] Rows: {before_rows} -> {after_rows}")
    return df


def filter_text(df: pd.DataFrame, column: str = "text", min_words: int = 5, max_words: int = 250) -> pd.DataFrame:
    """Filter and normalize text column based on word count constraints.

    Marks short texts as None (for rating-only analysis) and truncates long
    texts to meet word count bounds. Ensures text data quality for downstream
    NLP processing in the recommendation pipeline.

    Args:
        df (pd.DataFrame): Input dataframe with text column to filter.
        column (str, optional): Name of the text column to filter.
            Defaults to "text".
        min_words (int, optional): Minimum word count threshold. Texts below
            this are marked as None. Defaults to 5.
        max_words (int, optional): Maximum word count threshold. Texts exceeding
            this are truncated to this length. Defaults to 250.

    Returns:
        pd.DataFrame: Dataframe with filtered/normalized text column. Logs show
            counts of valid texts, short texts marked, and long texts truncated.

    Raises:
        ValueError: If the specified column does not exist in the dataframe.
    """
    _validate_columns(df, [column], required=True)
    word_count = df[column].str.split().str.len()
    short = (word_count < min_words).sum()
    long = (word_count > max_words).sum()
    df.loc[word_count < min_words, column] = None
    df.loc[word_count > max_words, column] = df.loc[word_count > max_words, column].str.split().str[:max_words].str.join(" ")
    valid_texts = (word_count >= min_words).sum()
    print(f"[INFO] Text filtering: {valid_texts} valid texts for NLP processing")
    print(f"[INFO] Marked {short} short texts as None (< {min_words} words)")
    print(f"[INFO] Truncated {long} long texts to {max_words} words")
    return df


def filter_high_signal_ratings(df: pd.DataFrame, min_rating: float = 4.0) -> pd.DataFrame:
    """Filter interactions to retain only high-quality positive feedback signals.

    Drops low-rating interactions (< min_rating) to ensure the dataset contains
    only quality implicit signals. Ratings >= min_rating are treated as positive
    user feedback for recommendation modeling.

    Args:
        df (pd.DataFrame): Input dataframe with rating column containing
            interaction ratings.
        min_rating (float, optional): Minimum rating threshold (inclusive).
            Only interactions with ratings >= this value are retained.
            Defaults to 4.0.

    Returns:
        pd.DataFrame: Filtered dataframe containing only high-signal ratings.
            Rows with ratings < min_rating are completely removed. Logs show
            row counts before and after filtering.

    Raises:
        ValueError: If the 'rating' column does not exist in the dataframe.
    """
    _validate_columns(df, ["rating"], required=True)
    before = len(df)
    df = df[df["rating"] >= min_rating].copy()
    after = len(df)
    removed = before - after
    print(f"[INFO] High-signal filtering (rating >= {min_rating}): {removed} rows removed")
    print(f"[INFO] Rows: {before} -> {after}")
    return df


def deduplicate_user_item(df: pd.DataFrame, user_col: str = "user_id", item_col: str = "parent_asin") -> pd.DataFrame:
    """Remove duplicate user-item interaction pairs.

    For implicit feedback scenarios, retains only the first occurrence of each
    unique user-item pair. Subsequent interactions from the same user for the
    same item are removed to ensure each pair represents a single signal.

    Args:
        df (pd.DataFrame): Input dataframe potentially containing duplicate
            user-item interactions.
        user_col (str, optional): Name of the user identifier column.
            Defaults to "user_id".
        item_col (str, optional): Name of the item identifier column.
            Defaults to "parent_asin".

    Returns:
        pd.DataFrame: Deduplicated dataframe with at most one interaction
            per user-item pair. Logs show duplicate counts and row changes.

    Raises:
        ValueError: If either user_col or item_col does not exist in the
            dataframe.
    """
    _validate_columns(df, [user_col, item_col], required=True)
    before = len(df)
    df = df.drop_duplicates(subset=[user_col, item_col], keep="first").copy()
    after = len(df)
    removed = before - after
    print(f"[INFO] Deduplication ('{user_col}' x '{item_col}'): {removed} duplicates removed")
    print(f"[INFO] Rows: {before} -> {after}")
    return df


def apply_iterative_k_core(df: pd.DataFrame, columns_config: Dict[str, int], max_iterations: int = 3) -> pd.DataFrame:
    """Apply iterative K-Core filtering on multiple dimensions simultaneously.

    Iteratively filters users and items in parallel until convergence. Each
    iteration removes users/items not meeting their k-core threshold, which may
    cause other users/items to fall below threshold in subsequent iterations.
    
    Produces a high-quality, densely-connected subgraph where both users and
    items have sufficient interaction counts.

    Args:
        df (pd.DataFrame): Input dataframe with user and item interaction data.
        columns_config (Dict[str, int]): Dictionary mapping column names to
            their k-core thresholds. Example: {'user_id': 10, 'parent_asin': 10}
            means keep users with > 10 interactions and items with > 10
            interactions.
        max_iterations (int, optional): Maximum number of iterations for
            convergence. Algorithm stops early if no rows are removed in an
            iteration (convergence achieved). Defaults to 3.

    Returns:
        pd.DataFrame: Filtered dataframe meeting all k-core constraints.
            Each user and item meets or exceeds their specified k-threshold.
            Logs show iteration-by-iteration removal counts and convergence
            status.

    Raises:
        ValueError: If any column specified in columns_config does not exist
            in the dataframe.
    """
    _validate_columns(df, list(columns_config.keys()), required=True)
    print(f"\n[INFO] Starting iterative K-Core filtering (max {max_iterations} iterations)")
    
    for iteration in range(max_iterations):
        before = len(df)
        for col, threshold in columns_config.items():
            counts = df[col].value_counts()
            valid_vals = counts[counts > threshold].index
            df = df[df[col].isin(valid_vals)].copy()
            removed = counts.shape[0] - len(valid_vals)
            remaining = len(valid_vals)
            print(f"    [Iter {iteration + 1}] {col} (k>{threshold}): {removed} removed | {remaining} remaining")
        
        after = len(df)
        rows_removed = before - after
        if rows_removed == 0:
            print(f"[INFO] K-Core convergence achieved at iteration {iteration + 1}")
            break
    else:
        print(f"[WARNING] K-Core did not converge after {max_iterations} iterations")
    
    print(f"[INFO] Final K-Core result: {len(df)} rows remaining\n")
    return df


def filter_top_n_items(df: pd.DataFrame, item_col: str = "parent_asin", top_n: int = 10000) -> pd.DataFrame:
    """Reduce dataset to interactions from only the Top-N most popular items.

    Keeps interactions only for the top_n items ranked by interaction count.
    Useful for focusing recommendation models on high-velocity catalog subsets
    and reducing computational complexity during training.

    Args:
        df (pd.DataFrame): Input dataframe with item interaction data.
        item_col (str, optional): Name of the item identifier column.
            Defaults to "parent_asin".
        top_n (int, optional): Number of top items to retain based on
            interaction count. All items outside top_n are removed along with
            their associated interactions. Defaults to 10000.

    Returns:
        pd.DataFrame: Filtered dataframe containing only interactions involving
            the top_n most popular items. Items outside top_n and all their
            interactions are removed. Logs show item and row count changes.

    Raises:
        ValueError: If the item_col does not exist in the dataframe.
    """
    _validate_columns(df, [item_col], required=True)
    item_counts = df[item_col].value_counts()
    top_items = item_counts.head(top_n).index
    before_items = len(item_counts)
    before_rows = len(df)
    df = df[df[item_col].isin(top_items)].copy()
    after_rows = len(df)
    items_removed = before_items - top_n
    print(f"[INFO] Top-N item filtering (top {top_n}): {items_removed} items removed")
    print(f"[INFO] Rows: {before_rows} -> {after_rows}")
    return df


def apply_confidence_weight(df: pd.DataFrame, verified_col: str = "verified_purchase", weight_col: str = "weight", verified_weight: float = 1.0, unverified_weight: float = 0.7) -> pd.DataFrame:
    """Assign confidence weights based on verified purchase status.

    Applies differential weighting to interactions based on verification status.
    Verified purchases receive higher weight, reflecting greater confidence in
    the interaction signal. Useful for implicit feedback modeling where signal
    reliability varies.

    Args:
        df (pd.DataFrame): Input dataframe with verified purchase indicator
            column.
        verified_col (str, optional): Name of the column indicating verified
            purchase status (values 0 or 1). Defaults to "verified_purchase".
        weight_col (str, optional): Name of the output weight column to create.
            Created or overwritten if already exists. Defaults to "weight".
        verified_weight (float, optional): Weight assigned to verified purchases.
            Defaults to 1.0.
        unverified_weight (float, optional): Weight assigned to unverified
            purchases. Typically lower than verified_weight to reduce their
            influence. Defaults to 0.7.

    Returns:
        pd.DataFrame: Dataframe with confidence weight column added/updated.
            Column values are numeric weights derived from verification status.

    Raises:
        ValueError: If the verified_col does not exist in the dataframe.
    """
    _validate_columns(df, [verified_col], required=True)
    df[weight_col] = df[verified_col].map({1: verified_weight, 0: unverified_weight})
    print(f"[INFO] Applied confidence weighting:")
    print(f"       Verified purchases: weight = {verified_weight}")
    print(f"       Unverified purchases: weight = {unverified_weight}")
    return df


def set_implicit_feedback_weight(df: pd.DataFrame, weight_col: str = "weight", interaction_weight: float = 1.0) -> pd.DataFrame:
    """Assign uniform weight to all interactions in implicit feedback scenario.

    Sets equal weight for all positive interactions when no confidence signal
    differentiates them. Suitable for implicit feedback contexts where all
    interactions represent equal strength signals.

    Args:
        df (pd.DataFrame): Input dataframe to add or update weight column.
        weight_col (str, optional): Name of the weight column to create/update.
            If column already exists, values are overwritten. Defaults to
            "weight".
        interaction_weight (float, optional): Uniform weight value assigned to
            all interactions. Typically 1.0 for implicit feedback. Defaults to
            1.0.

    Returns:
        pd.DataFrame: Dataframe with uniform weight column added/updated.
            All rows contain the same weight value for equal implicit signal
            strength.
    """
    df[weight_col] = interaction_weight
    print(f"[INFO] Set implicit feedback weight: all interactions = {interaction_weight}")
    return df


def encode_labels(df: pd.DataFrame, columns: List[str], encoders: Optional[Dict[str, LabelEncoder]] = None) -> Tuple[pd.DataFrame, Dict[str, LabelEncoder]]:
    """Encode categorical columns to integer labels for machine learning models.

    On training data, fits new LabelEncoders for specified columns. On
    validation/test data, applies pre-fitted encoders to ensure consistent
    label mapping across all splits. Prevents label encoding leakage between
    train and test sets.

    Args:
        df (pd.DataFrame): Input dataframe with categorical columns to encode.
        columns (List[str]): List of column names to encode. All specified
            columns must exist; missing columns raise ValueError.
        encoders (Optional[Dict[str, LabelEncoder]], optional): Dictionary
            mapping column names to pre-fitted LabelEncoder instances. If None,
            new encoders are fit on the input data (training mode). If provided,
            pre-fitted encoders are applied to ensure consistent transformation
            (test/validation mode). Defaults to None.

    Returns:
        Tuple[pd.DataFrame, Dict[str, LabelEncoder]]: Tuple containing:
            - Encoded dataframe with columns converted to integer labels
            - Dictionary of fitted encoders (newly fit if encoders=None, or
              the provided encoders dict)

    Raises:
        ValueError: If any specified column does not exist in the dataframe.
    """
    columns = _validate_columns(df, columns, required=True)
    if encoders is None:
        encoders = {}
    for column in columns:
        if column not in encoders:
            encoders[column] = LabelEncoder()
            df[column] = encoders[column].fit_transform(df[column])
            print(f"[INFO] Encoded '{column}' (fitted) -> {df[column].nunique()} unique labels")
        else:
            df[column] = encoders[column].transform(df[column])
            print(f"[INFO] Encoded '{column}' (applied) -> {df[column].nunique()} unique labels")
    return df, encoders


def time_based_split(df: pd.DataFrame, column: str = "timestamp", val_year: int = 2021, test_year: int = 2022) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Split dataframe into train, validation, and test sets using time ordering.

    Performs temporal splitting to prevent data leakage and ensure realistic
    evaluation. Training data predates validation data, which predates test data.
    Suitable for time-series interactions like reviews, clicks, and ratings.

    Args:
        df (pd.DataFrame): Input dataframe with interaction timestamps to split.
        column (str, optional): Name of the datetime column for splitting.
            Must contain datetime values extracted via dt.year. Defaults to
            "timestamp".
        val_year (int, optional): Starting year (inclusive) for validation set.
            Validation includes all data from val_year up to (but not including)
            test_year. Defaults to 2021.
        test_year (int, optional): Starting year (inclusive) for test set.
            Test includes all data from test_year onward. Defaults to 2022.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]: Tuple containing:
            - train_df: Data with year < val_year
            - val_df: Data with year >= val_year and year < test_year
            - test_df: Data with year >= test_year

    Raises:
        ValueError: If the specified column does not exist in the dataframe.
    """
    _validate_columns(df, [column], required=True)
    train_df = df[df[column].dt.year < val_year].copy()
    val_df = df[(df[column].dt.year >= val_year) & (df[column].dt.year < test_year)].copy()
    test_df = df[df[column].dt.year >= test_year].copy()
    total = len(df)
    print(f"[INFO] Time-based split:")
    print(f"       Train: {len(train_df)} rows ({len(train_df)/total*100:.1f}%)")
    print(f"       Val:   {len(val_df)} rows ({len(val_df)/total*100:.1f}%)")
    print(f"       Test:  {len(test_df)} rows ({len(test_df)/total*100:.1f}%)")
    return train_df, val_df, test_df


def normalize(df: pd.DataFrame, column: str, scaler: Optional[MinMaxScaler] = None) -> Tuple[pd.DataFrame, MinMaxScaler]:
    """Normalize a single column to [0, 1] range using MinMaxScaler.

    On training data, fits MinMaxScaler to column values to learn min/max bounds.
    On validation/test data, applies pre-fitted scaler to ensure consistent
    normalization using training statistics. Prevents normalization leakage
    between train and test sets.

    Args:
        df (pd.DataFrame): Input dataframe with column to normalize.
        column (str): Name of the column to normalize to [0, 1] range.
            Must exist and contain numeric values.
        scaler (Optional[MinMaxScaler], optional): Pre-fitted MinMaxScaler
            instance trained on training data. If None, a new scaler is fit
            on the input data (training mode). If provided, the pre-fitted
            scaler is applied for consistent transformation (test/validation
            mode). Defaults to None.

    Returns:
        Tuple[pd.DataFrame, MinMaxScaler]: Tuple containing:
            - Normalized dataframe with specified column scaled to [0, 1]
            - MinMaxScaler instance (newly fit if scaler=None, or the
              provided scaler)

    Raises:
        ValueError: If the specified column does not exist in the dataframe.
    """
    _validate_columns(df, [column], required=True)
    if scaler is None:
        scaler = MinMaxScaler()
        df[column] = scaler.fit_transform(df[[column]])
        print(f"[INFO] Normalized '{column}' (fitted) -> range [0, 1]")
    else:
        df[column] = scaler.transform(df[[column]])
        print(f"[INFO] Normalized '{column}' (applied) -> range [0, 1]")
    return df, scaler
