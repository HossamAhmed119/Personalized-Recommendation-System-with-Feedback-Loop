"""Statistical summary utilities for exploratory data analysis.

This module provides functions to compute and summarize key statistics
about datasets, including missing values, distributions, sparsity, activity
patterns, and long-tail detection.

All functions return dictionaries with clear keys for easy reporting
and integration into analysis notebooks.
"""

from typing import Dict, Any, Tuple
import pandas as pd
import numpy as np


def summarize_missing_values(df: pd.DataFrame) -> Dict[str, Any]:
    """Calculate missing value statistics for all columns.

    Computes count and percentage of missing values per column, useful for
    understanding data quality and completeness.

    Args:
        df (pd.DataFrame): Input dataframe to analyze.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'total_rows': Total number of rows
            - 'columns_missing': Dict mapping column names to {'count': int, 'percentage': float}
            - 'columns_complete': List of columns with no missing values
            - 'overall_completeness': Float between 0-100 for overall data completeness

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'A': [1, None, 3], 'B': [4, 5, 6]})
        >>> summary = summarize_missing_values(df)
        >>> summary['columns_missing']
        {'A': {'count': 1, 'percentage': 33.33}}
    """
    total_rows = len(df)
    missing_info = {}

    for col in df.columns:
        missing_count = df[col].isna().sum()
        if missing_count > 0:
            missing_info[col] = {
                'count': int(missing_count),
                'percentage': round((missing_count / total_rows) * 100, 2)
            }

    complete_cols = [col for col in df.columns if col not in missing_info]

    total_cells = total_rows * len(df.columns)
    total_missing = sum(info['count'] for info in missing_info.values())
    overall_completeness = round(((total_cells - total_missing) / total_cells) * 100, 2)

    return {
        'total_rows': total_rows,
        'columns_missing': missing_info,
        'columns_complete': complete_cols,
        'overall_completeness': overall_completeness
    }


def summarize_distribution(df: pd.DataFrame, column: str) -> Dict[str, Any]:
    """Calculate distribution statistics for a numeric column, including missing data context.

    Provides descriptive statistics including mean, median, std, quartiles,
    min, max, skewness, and the percentage of missing values.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to analyze (must be numeric).

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'column': Column name analyzed
            - 'missing_count': Number of missing/NaN values
            - 'missing_percentage': Percentage of missing values
            - 'count': Number of non-null values
            - 'mean': Arithmetic mean
            - 'median': 50th percentile
            - 'std': Standard deviation
            - 'min': Minimum value
            - 'q25': 25th percentile
            - 'q75': 75th percentile
            - 'max': Maximum value
            - 'skewness': Fisher-Pearson skewness coefficient
            - 'kurtosis': Excess kurtosis

    Raises:
        KeyError: If column does not exist in dataframe.
        ValueError: If column is not numeric.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({'price': [10.5, np.nan, 30.0, 45.5, np.nan]})
        >>> dist = summarize_distribution(df, 'price')
        >>> dist['missing_percentage']
        40.0
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    if not pd.api.types.is_numeric_dtype(df[column]):
        raise ValueError(f"Column '{column}' must be numeric")

    missing_count = int(df[column].isna().sum())
    missing_pct = round((missing_count / len(df)) * 100, 2)

    data = df[column].dropna()

    if data.empty:
        return {
            'column': column,
            'missing_count': missing_count,
            'missing_percentage': missing_pct,
            'count': 0,
            'error': 'All values are NaN'
        }

    return {
        'column': column,
        'missing_count': missing_count,
        'missing_percentage': missing_pct,
        'count': len(data),
        'mean': round(data.mean(), 4),
        'median': round(data.median(), 4),
        'std': round(data.std(), 4),
        'min': round(data.min(), 4),
        'q25': round(data.quantile(0.25), 4),
        'q75': round(data.quantile(0.75), 4),
        'max': round(data.max(), 4),
        'skewness': round(data.skew(), 4),
        'kurtosis': round(data.kurtosis(), 4)
    }


def calculate_sparsity(
    df: pd.DataFrame,
    user_col: str = 'user_id',
    item_col: str = 'parent_asin'
) -> Dict[str, Any]:
    """Calculate sparsity of user-item interaction matrix.

    Computes matrix sparsity as the ratio of missing interactions to
    total possible interactions. Useful for understanding coverage in
    recommender systems.

    Args:
        df (pd.DataFrame): Input dataframe with user and item columns.
        user_col (str, optional): Column name for users. Defaults to 'user_id'.
        item_col (str, optional): Column name for items. Defaults to 'parent_asin'.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'n_users': Number of unique users
            - 'n_items': Number of unique items
            - 'n_interactions': Total number of interactions
            - 'possible_interactions': Maximum possible interactions (n_users * n_items)
            - 'sparsity_percentage': Sparsity as percentage (0-100)
            - 'density_percentage': Density as percentage (complement of sparsity)

    Raises:
        KeyError: If either column does not exist in dataframe.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 1, 2, 3],
        ...     'parent_asin': ['A', 'B', 'A', 'C']
        ... })
        >>> sparsity = calculate_sparsity(df)
        >>> sparsity['sparsity_percentage']  # 8 out of 12 possible = 66.67% sparse
        66.67
    """
    if user_col not in df.columns or item_col not in df.columns:
        raise KeyError(f"Columns '{user_col}' and/or '{item_col}' not found")

    n_users = df[user_col].nunique()
    n_items = df[item_col].nunique()
    n_interactions = len(df)
    possible_interactions = n_users * n_items

    sparsity_percentage = round(
        (1 - (n_interactions / possible_interactions)) * 100,
        2
    ) if possible_interactions > 0 else 0.0

    return {
        'n_users': n_users,
        'n_items': n_items,
        'n_interactions': n_interactions,
        'possible_interactions': possible_interactions,
        'sparsity_percentage': sparsity_percentage,
        'density_percentage': round(100 - sparsity_percentage, 2)
    }


def summarize_activity(
    df: pd.DataFrame,
    group_col: str = 'user_id',
    count_col: str = 'parent_asin'
) -> Dict[str, Any]:
    """Calculate activity statistics grouped by a column.

    Provides per-group activity metrics including count, mean, median,
    min, and max interactions per group. Useful for user/item segmentation.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_col (str, optional): Column to group by (e.g., 'user_id').
            Defaults to 'user_id'.
        count_col (str, optional): Column to count per group. Defaults to 'parent_asin'.

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'group_col': Name of grouping column
            - 'n_groups': Number of unique groups
            - 'mean_activity': Average count per group
            - 'median_activity': Median count per group
            - 'std_activity': Standard deviation of counts
            - 'min_activity': Minimum count in any group
            - 'max_activity': Maximum count in any group
            - 'total_interactions': Total count across all groups

    Raises:
        KeyError: If either column does not exist in dataframe.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'user_id': [1, 1, 1, 2, 2, 3],
        ...     'parent_asin': ['A', 'B', 'C', 'A', 'D', 'B']
        ... })
        >>> activity = summarize_activity(df, 'user_id', 'parent_asin')
        >>> activity['mean_activity']
        2.0
    """
    if group_col not in df.columns or count_col not in df.columns:
        raise KeyError(f"Columns '{group_col}' and/or '{count_col}' not found")

    group_counts = df.groupby(group_col).size()

    return {
        'group_col': group_col,
        'n_groups': len(group_counts),
        'mean_activity': round(group_counts.mean(), 2),
        'median_activity': round(group_counts.median(), 2),
        'std_activity': round(group_counts.std(), 2),
        'min_activity': int(group_counts.min()),
        'max_activity': int(group_counts.max()),
        'total_interactions': int(group_counts.sum())
    }


def detect_long_tail(
    df: pd.DataFrame,
    group_col: str = 'parent_asin',
    threshold_percentile: int = 80
) -> Dict[str, Any]:
    """Detect long-tail distribution in grouped data.

    Identifies head vs tail split based on cumulative percentage threshold.
    Useful for understanding Pareto principle in user/item distributions.

    Args:
        df (pd.DataFrame): Input dataframe.
        group_col (str, optional): Column to group by (e.g., 'parent_asin').
            Defaults to 'parent_asin'.
        threshold_percentile (int, optional): Cumulative percentage threshold
            for head vs tail split. Defaults to 80 (80/20 rule).

    Returns:
        Dict[str, Any]: Dictionary with keys:
            - 'group_col': Name of grouping column
            - 'total_groups': Total number of groups
            - 'head_groups': Number of groups in head
            - 'tail_groups': Number of groups in tail
            - 'head_percentage': Percentage of groups in head
            - 'tail_percentage': Percentage of groups in tail
            - 'head_interactions': Total interactions in head groups
            - 'tail_interactions': Total interactions in tail groups
            - 'head_coverage_percentage': Percentage of total interactions in head

    Raises:
        KeyError: If column does not exist in dataframe.
        ValueError: If threshold_percentile not between 0 and 100.

    Example:
        >>> import pandas as pd
        >>> df = pd.DataFrame({
        ...     'parent_asin': ['A']*100 + ['B']*50 + ['C']*10 + ['D']*5
        ... })
        >>> long_tail = detect_long_tail(df, 'parent_asin', threshold_percentile=80)
        >>> long_tail['head_percentage']
        50.0  # 2 items (A, B) are 80% of interactions
    """
    if group_col not in df.columns:
        raise KeyError(f"Column '{group_col}' not found in dataframe")

    if not (0 < threshold_percentile < 100):
        raise ValueError("threshold_percentile must be between 0 and 100")

    group_counts = df.groupby(group_col).size().sort_values(ascending=False)
    total_interactions = group_counts.sum()
    cumsum_percentage = (group_counts.cumsum() / total_interactions * 100)

    # Fixed Edge Case: Ensure head is never 0 even if the first item exceeds the threshold
    head_threshold_idx = (cumsum_percentage <= threshold_percentile).sum()
    head_groups = max(1, int(head_threshold_idx))
    
    tail_groups = len(group_counts) - head_groups

    head_interactions = group_counts.iloc[:head_groups].sum() if head_groups > 0 else 0
    tail_interactions = group_counts.iloc[head_groups:].sum() if tail_groups > 0 else 0

    head_percentage = round((head_groups / len(group_counts)) * 100, 2)
    tail_percentage = round(100 - head_percentage, 2)
    head_coverage = round((head_interactions / total_interactions) * 100, 2)

    print(f"[INFO] Long-tail analysis: {head_groups} items ({head_percentage}%) "
          f"account for {head_coverage}% of interactions")

    return {
        'group_col': group_col,
        'total_groups': len(group_counts),
        'head_groups': head_groups,
        'tail_groups': tail_groups,
        'head_percentage': head_percentage,
        'tail_percentage': tail_percentage,
        'head_interactions': int(head_interactions),
        'tail_interactions': int(tail_interactions),
        'head_coverage_percentage': head_coverage
    }