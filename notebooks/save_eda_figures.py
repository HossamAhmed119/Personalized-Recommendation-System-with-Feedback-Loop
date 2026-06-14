"""
Save EDA Figures Script
Run this from the project root:
    python notebooks/save_eda_figures.py
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

# Get the notebook directory and go to project root
notebook_dir = Path(__file__).parent.absolute()
project_root = notebook_dir.parent.absolute()
os.chdir(project_root)
sys.path.insert(0, str(project_root))

from src.utils.config_loader import load_config

# ── Config & Data ──────────────────────────────────────────────
config   = load_config("configs/data_config.yaml")
raw_path = config['paths']['raw_data']
fig_path = "docs/figures/"
os.makedirs(fig_path, exist_ok=True)

# Load data from parquet instead of CSV
df_ratings = pd.read_parquet(raw_path + "Electronics_ratings.parquet")

# Ensure timestamp is datetime
if 'timestamp' in df_ratings.columns:
    df_ratings['timestamp'] = pd.to_datetime(df_ratings['timestamp'], unit='ms', errors='coerce')
if 'verified_purchase' in df_ratings.columns:
    df_ratings['verified_purchase'] = df_ratings['verified_purchase'].astype(int)

print("[INFO] Data loaded ✅")


# ── Helper ─────────────────────────────────────────────────────
def save(name):
    path = os.path.join(fig_path, name)
    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    plt.close()
    print(f"[SAVED] {path}")


# ── 1. Rating Distribution ─────────────────────────────────────
plt.figure(figsize=(10, 5))

plt.subplot(1, 2, 1)
df_ratings['rating'].value_counts().sort_index().plot(kind='bar', color='steelblue')
plt.title('Rating Distribution')
plt.xlabel('Rating')
plt.ylabel('Count')

plt.subplot(1, 2, 2)
df_ratings['rating'].value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('Rating Percentage')

save("01_rating_distribution.png")


# ── 2. User Activity Distribution ─────────────────────────────
user_activity = df_ratings['user_id'].value_counts()

plt.figure(figsize=(6, 4))
plt.hist(user_activity, bins=50)
plt.title("Distribution of Number of Ratings Per User")
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Users')
save("02_user_activity_distribution.png")


# ── 3. Box Plots ───────────────────────────────────────────────
plt.figure(figsize=(14, 6))

plt.subplot(1, 2, 1)
plt.boxplot(df_ratings['rating'], vert=False,
            flierprops=dict(marker='o', markersize=3))
plt.title('Rating BoxPlot')
plt.xlabel('Values')
plt.ylabel('Rating')

plt.subplot(1, 2, 2)
plt.boxplot(df_ratings['helpful_vote'], vert=False,
            flierprops=dict(marker='o', markersize=3))
plt.title('Helpful Vote BoxPlot')
plt.xlabel('Values')
plt.ylabel('Helpful Votes')

save("03_boxplots.png")


# ── 4. Time Analysis ───────────────────────────────────────────
time_curve = df_ratings['timestamp'].dt.year.value_counts().sort_index()

plt.figure(figsize=(10, 5))
plt.plot(time_curve.index, time_curve.values, marker='o')
plt.title('Reviews Over Time')
plt.xlabel('Year')
plt.ylabel('Number of Reviews')
plt.axvline(x=2020, color='r', linestyle='--', label='COVID-19 Peak')
plt.legend()
save("04_time_analysis.png")


# ── 5. Item Activity Distribution ─────────────────────────────
item_activity = df_ratings['asin'].value_counts()

plt.figure(figsize=(6, 4))
plt.hist(item_activity, bins=50)
plt.title("Distribution of Number of Ratings Per Item")
plt.xlabel('Number of Ratings')
plt.ylabel('Number of Items')
save("05_item_activity_distribution.png")


# ── 6. Top 10 Most Reviewed Items ─────────────────────────────
top_10 = item_activity.head(10)

plt.figure(figsize=(10, 5))
sns.barplot(x=top_10.index, y=top_10.values)
plt.title('Top 10 Most Reviewed Items')
plt.xlabel('Item ID')
plt.ylabel('Number of Reviews')
plt.xticks(rotation=45)
save("06_top10_items.png")


# ── 7. Correlation Heatmap ─────────────────────────────────────
correlation = df_ratings[["rating", "helpful_vote", "verified_purchase"]].corr()

plt.figure(figsize=(8, 6))
sns.heatmap(correlation, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
save("07_correlation_heatmap.png")


# ── 8. Verified vs Unverified ──────────────────────────────────
plt.figure(figsize=(6, 4))
df_ratings.groupby('verified_purchase')['rating'].mean().plot(kind='bar')
plt.title('Average Rating: Verified vs Unverified')
plt.xlabel('Verified Purchase')
plt.ylabel('Average Rating')
save("08_verified_vs_unverified.png")


# ── 9. Text Length Distribution ────────────────────────────────
text_words = df_ratings['text'].dropna().str.split().str.len()

plt.figure(figsize=(10, 5))
plt.hist(text_words, bins=50)
plt.title('Text Length By Word')
plt.xlabel('Number of Words')
plt.ylabel('Number of Reviews')
plt.xlim(0, 500)
save("09_text_length_distribution.png")


# ── 10. User Segmentation ──────────────────────────────────────
user_counts  = df_ratings['user_id'].value_counts()
user_activity_col = df_ratings['user_id'].map(user_counts)
user_segment = pd.cut(user_activity_col,
                      bins=[0, 2, 10, np.inf],
                      labels=['Light', 'Medium', 'Heavy'])

plt.figure(figsize=(6, 6))
user_segment.value_counts().plot(kind='pie', autopct='%1.1f%%')
plt.title('User Segmentation')
save("10_user_segmentation.png")


# ── 11. Spam Detection ─────────────────────────────────────────
df_sorted = df_ratings.sort_values(['user_id', 'timestamp'])
df_sorted['time_diff'] = df_sorted.groupby('user_id')['timestamp'].diff().dt.total_seconds()

user_spam_check = df_sorted.groupby('user_id').agg({
    'rating': 'count',
    'time_diff': 'min'
}).rename(columns={'rating': 'total_ratings', 'time_diff': 'min_time_gap'})

plt.figure(figsize=(12, 6))
sns.scatterplot(data=user_spam_check, x='total_ratings', y='min_time_gap', alpha=0.6)
plt.axhline(y=10, color='r', linestyle='--', label='Possible Bot (10s gap)')
plt.yscale('log')
plt.title('Spam Detection: Ratings Count vs Min Time Gap')
plt.legend()
save("11_spam_detection.png")


# ── 12. Yearly Ratings Distribution ────────────────────────────
yearly_counts = df_ratings['timestamp'].dt.year.value_counts().sort_index()

plt.figure(figsize=(12, 6))
sns.barplot(x=yearly_counts.index.astype(int), y=yearly_counts.values, palette='viridis')
plt.title('Yearly Ratings Distribution')
plt.xlabel('Year')
plt.ylabel('Number of Ratings')
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
save("12_yearly_distribution.png")


print(f"\n✅ All figures saved to: {fig_path}")
