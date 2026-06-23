"""Visualization utilities for exploratory data analysis.

This module provides reusable plotting functions for EDA tasks including
bar/pie charts, histograms, boxplots, line plots, heatmaps, and scatter plots.
All functions support optional figure saving to the docs/figures/ directory.

Functions are framework-agnostic and accept flexible parameters for maximum
reusability across different datasets and analysis contexts.
"""

from typing import Optional, Tuple, List
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def _ensure_figure_dir(save_path: str) -> None:
    """Create figure directory if it doesn't exist.
    
    Args:
        save_path (str): Path where figure will be saved.
    """
    os.makedirs(os.path.dirname(save_path), exist_ok=True)


def plot_bar_distribution(
    df: pd.DataFrame,
    column: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Count",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    pre_aggregated: bool = False,
    value_col: Optional[str] = None
) -> None:
    """Create a bar chart for categorical/discrete distributions.

    Plots value counts in a bar chart, useful for rating distributions,
    user segments, store tiers, etc. Can also plot already-aggregated data
    (e.g., Top N items already counted).

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to visualize (categories or labels).
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "Count".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        save_path (Optional[str], optional): Path to save figure. Defaults to None.
        pre_aggregated (bool, optional): If True, treats `column` as labels
            and `value_col` as the already-computed counts/values, skipping
            internal value_counts(). Defaults to False.
        value_col (Optional[str], optional): Required if pre_aggregated=True.
            Column containing the numeric values to plot directly.

    Raises:
        KeyError: If column does not exist in dataframe.
        ValueError: If pre_aggregated=True but value_col not provided.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    plt.figure(figsize=figsize)

    if pre_aggregated:
        if value_col is None or value_col not in df.columns:
            raise ValueError("value_col must be provided and exist in dataframe when pre_aggregated=True")
        plot_data = df.set_index(column)[value_col]
    else:
        plot_data = df[column].value_counts()
        if pd.api.types.is_numeric_dtype(df[column]):
            plot_data = plot_data.sort_index()

    plot_data.plot(kind='bar', color='steelblue', edgecolor='black')
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_pie_distribution(
    df: pd.DataFrame,
    column: str,
    title: str = "",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    top_n: Optional[int] = None
) -> None:
    """Create a pie chart for categorical distributions.

    Visualizes proportions of categories. Useful for rating segments,
    user classifications, etc.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name to visualize. Must be categorical.
        title (str, optional): Chart title. Defaults to "".
        figsize (Tuple[int, int], optional): Figure size (width, height).
            Defaults to (10, 8).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.
        top_n (Optional[int], optional): If specified, show only top N
            categories and group rest as 'Other'. Defaults to None (show all).

    Raises:
        KeyError: If column does not exist in dataframe.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    value_counts = df[column].value_counts()
    
    # Auto-grouping edge case for high unique counts to prevent crashes
    if top_n is None and len(value_counts) > 20:
        print(f"[WARNING] {len(value_counts)} unique categories found in '{column}'. Auto-grouping into Top 15 + 'Other'.")
        top_n = 15

    if top_n and len(value_counts) > top_n:
        top_categories = value_counts.head(top_n)
        other_count = value_counts.iloc[top_n:].sum()
        value_counts = pd.concat([top_categories, pd.Series({'Other': other_count})])

    plt.figure(figsize=figsize)
    plt.pie(
        value_counts,
        labels=value_counts.index,
        autopct='%1.1f%%',
        startangle=90,
        colors=sns.color_palette("husl", len(value_counts))
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_histogram(
    df: pd.DataFrame,
    column: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "Frequency",
    bins: int = 30,
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    log_scale: bool = False
) -> None:
    """Create a histogram for continuous/discrete numeric distributions.

    Visualizes frequency distributions, useful for user activity,
    item activity, text length, etc.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name (must be numeric).
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "Frequency".
        bins (int, optional): Number of histogram bins. Defaults to 30.
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.
        log_scale (bool, optional): If True, use log scale for y-axis.
            Useful for long-tail distributions. Defaults to False.

    Raises:
        KeyError: If column does not exist in dataframe.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    plt.figure(figsize=figsize)
    plt.hist(df[column].dropna(), bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    if log_scale:
        plt.yscale('log')
    plt.grid(axis='y', alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_boxplot(
    df: pd.DataFrame,
    column: str,
    title: str = "",
    xlabel: str = "",
    figsize: Tuple[int, int] = (10, 4),
    save_path: Optional[str] = None
) -> None:
    """Create a horizontal boxplot for outlier detection and distribution shape.

    Visualizes median, quartiles, and outliers.

    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): Column name (must be numeric).
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 4).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.

    Raises:
        KeyError: If column does not exist in dataframe.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    plt.figure(figsize=figsize)
    data = df[column].dropna()
    
    # Restored to horizontal (vert=False) with specific flierprops
    plt.boxplot(data, vert=False, patch_artist=True, 
                boxprops=dict(facecolor='lightblue', color='steelblue'),
                flierprops=dict(marker='o', markersize=3, alpha=0.5, color='red'))
                
    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.grid(axis='x', linestyle='--', alpha=0.7)
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_lineplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (12, 6),
    save_path: Optional[str] = None,
    marker_year: Optional[int] = None,
    marker_label: str = ""
) -> None:
    """Create a line plot for time-series analysis.

    Visualizes trends over time with optional vertical markers for events.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_col (str): Column name for x-axis (typically time/year).
        y_col (str): Column name for y-axis (values to plot).
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (12, 6).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.
        marker_year (Optional[int], optional): Year to mark with vertical line
            (e.g., 2020 for COVID-19). Defaults to None.
        marker_label (str, optional): Label for marker line. Defaults to "".

    Raises:
        KeyError: If either column does not exist in dataframe.
    """
    if x_col not in df.columns or y_col not in df.columns:
        raise KeyError(f"Columns {x_col} and/or {y_col} not found in dataframe")

    plt.figure(figsize=figsize)
    plt.plot(df[x_col], df[y_col], marker='o', linewidth=2, color='steelblue')

    if marker_year is not None:
        plt.axvline(x=marker_year, color='red', linestyle='--', linewidth=2, label=marker_label)
        plt.legend()

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_heatmap(
    df: pd.DataFrame,
    title: str = "",
    figsize: Tuple[int, int] = (10, 8),
    save_path: Optional[str] = None,
    cmap: str = "coolwarm"
) -> None:
    """Create a correlation heatmap for numeric columns.

    Visualizes correlation matrix as a heatmap.

    Args:
        df (pd.DataFrame): Input dataframe (should contain numeric columns).
        title (str, optional): Chart title. Defaults to "".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 8).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.
        cmap (str, optional): Colormap name. Defaults to "coolwarm".

    Raises:
        ValueError: If dataframe has no numeric columns.
    """
    numeric_df = df.select_dtypes(include=[np.number])
    if numeric_df.empty:
        raise ValueError("Dataframe contains no numeric columns for correlation")

    corr_matrix = numeric_df.corr()

    plt.figure(figsize=figsize)
    sns.heatmap(
        corr_matrix,
        annot=True,
        fmt='.2f',
        cmap=cmap,
        center=0,
        cbar_kws={'label': 'Correlation'},
        square=True
    )
    plt.title(title, fontsize=14, fontweight='bold')
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_scatterplot(
    df: pd.DataFrame,
    x_col: str,
    y_col: str,
    title: str = "",
    xlabel: str = "",
    ylabel: str = "",
    figsize: Tuple[int, int] = (10, 6),
    save_path: Optional[str] = None,
    threshold_x: Optional[float] = None,
    threshold_y: Optional[float] = None,
    threshold_label: str = "",
    log_scale_y: bool = False       # ← جديد
) -> None:
    """Create a scatter plot for relationship and anomaly analysis.

    Visualizes relationships between two numeric variables with optional
    threshold lines for anomaly detection and optional log scale on Y axis.

    Args:
        df (pd.DataFrame): Input dataframe.
        x_col (str): Column name for x-axis.
        y_col (str): Column name for y-axis.
        title (str, optional): Chart title. Defaults to "".
        xlabel (str, optional): X-axis label. Defaults to "".
        ylabel (str, optional): Y-axis label. Defaults to "".
        figsize (Tuple[int, int], optional): Figure size. Defaults to (10, 6).
        save_path (Optional[str], optional): Path to save figure.
            Defaults to None.
        threshold_x (Optional[float], optional): Vertical line at this x-value
            (e.g., spam threshold). Defaults to None.
        threshold_y (Optional[float], optional): Horizontal line at this y-value.
            Defaults to None.
        threshold_label (str, optional): Label for threshold lines. Defaults to "".
        log_scale_y (bool, optional): If True, applies log scale to Y axis.
            Useful when data has extreme outliers (e.g., time gaps in spam detection).
            Defaults to False.

    Raises:
        KeyError: If either column does not exist in dataframe.
    """
    plt.figure(figsize=figsize)
    plt.scatter(df[x_col], df[y_col], alpha=0.6, s=50, 
                color='steelblue', edgecolors='black', linewidth=0.5)

    if threshold_x is not None:
        plt.axvline(x=threshold_x, color='red', linestyle='--', 
                    linewidth=2, label=threshold_label)
    if threshold_y is not None:
        plt.axhline(y=threshold_y, color='red', linestyle='--', 
                    linewidth=2, label=threshold_label)

    if threshold_x is not None or threshold_y is not None:
        plt.legend()

    if log_scale_y:                  # ← جديد
        plt.yscale('log')

    plt.title(title, fontsize=14, fontweight='bold')
    plt.xlabel(xlabel, fontsize=12)
    plt.ylabel(ylabel, fontsize=12)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")

    plt.show()


def plot_categorical_overview(
    df: pd.DataFrame, 
    column: str, 
    title: str, 
    top_n: int = 10, 
    save_path: Optional[str] = None,
    sort_by_index: bool = False
) -> None:
    """Generates a multi-panel side-by-side view (Bar + Pie) for categorical data.
    
    Args:
        df (pd.DataFrame): Input dataframe.
        column (str): The categorical column to visualize.
        title (str): The main title for the figure.
        top_n (int): Number of top categories to display before grouping into 'Other'. Defaults to 10.
        save_path (Optional[str], optional): Path to save figure. Defaults to None.
        sort_by_index (bool, optional): If True, sorts categories by their natural
            order (e.g., 1,2,3,4,5) instead of by frequency. Useful for ordinal data
            like ratings. Defaults to False.
        
    Raises:
        KeyError: If column does not exist in dataframe.
    """
    if column not in df.columns:
        raise KeyError(f"Column '{column}' not found in dataframe")

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    fig.suptitle(title, fontsize=16, y=1.05, fontweight='bold')
    
    value_counts = df[column].value_counts(dropna=False)

    if sort_by_index:
        value_counts = value_counts.sort_index()  # ← لازم تتعمل قبل أي حاجة تانية

    if len(value_counts) > top_n:
        if sort_by_index:
            plot_data = value_counts.iloc[:top_n]
        else:
            top_categories = value_counts.iloc[:top_n]
            others_count = value_counts.iloc[top_n:].sum()
            plot_data = pd.concat([top_categories, pd.Series({'Other': others_count})])
    else:
        plot_data = value_counts

    # 1. Bar Chart (Left Panel)
    sns.barplot(x=plot_data.index.astype(str), y=plot_data.values, palette="viridis", ax=axes[0])
    axes[0].set_title(f"Top {top_n} {column} (Bar)", fontsize=13)
    axes[0].set_ylabel("Count")
    axes[0].tick_params(axis='x', rotation=45)
    axes[0].grid(axis='y', linestyle='--', alpha=0.7)

    # 2. Pie Chart (Right Panel)
    axes[1].pie(
        plot_data.values, 
        labels=plot_data.index.astype(str), 
        autopct='%1.1f%%', 
        startangle=140, 
        colors=sns.color_palette("husl", len(plot_data))
    )
    axes[1].set_title(f"Top {top_n} {column} (Pie)", fontsize=13)

    plt.tight_layout()
    
    if save_path:
        _ensure_figure_dir(save_path)
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"[INFO] Figure saved to {save_path}")
        
    plt.show()