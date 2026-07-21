"""
End-to-End Preprocessing Pipeline for Large Amazon Datasets
Location: src/models/run_preprocessing.py
"""

import sys
import os
import gc
from pathlib import Path

# ============================================
# DYNAMIC PATH RESOLUTION
# ============================================
current_script_path = Path(__file__).resolve()
PROJECT_ROOT = current_script_path.parent.parent.parent

os.chdir(PROJECT_ROOT)
sys.path.append(str(PROJECT_ROOT))

# ============================================
# IMPORTS
# ============================================
import yaml
import logging
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from datetime import datetime
import joblib
from sklearn.preprocessing import MinMaxScaler
import warnings

warnings.filterwarnings('ignore')

# Project imports
from src.data_pipeline.preprocess import *
from src.data_pipeline.features import *
from src.utils.eda_summary import *

# ============================================
# LOGGING SETUP
# ============================================
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# ============================================
# MEMORY-OPTIMIZED FUNCTIONS
# ============================================
def memory_optimized_detect_spam(df, max_reviews_per_day=5, min_time_gap=10):
    """
    Memory-efficient spam detection tailored for 20M+ rows.
    """
    logger.info("    Extracting dates for spam validation...")
    dates = pd.to_datetime(df['timestamp']).dt.date
    
    logger.info("    Finding high-frequency spam users...")
    user_day_counts = df.groupby(['user_id', dates]).size().reset_index(name='count')
    spam_users_by_day = user_day_counts[user_day_counts['count'] > max_reviews_per_day]['user_id'].unique()
    
    logger.info("    Filtering out spam users from dataset...")
    spam_mask = df['user_id'].isin(spam_users_by_day)
    df_cleaned = df[~spam_mask].copy()
    
    logger.info(f"    Removed {len(df) - len(df_cleaned):,} rows from high-frequency spam users.")
    
    del user_day_counts, spam_users_by_day, spam_mask
    gc.collect()
    
    return df_cleaned

def generate_report(train_df, val_df, test_df, meta_df, start_time):
    duration = datetime.now() - start_time
    report = []
    report.append("=" * 60)
    report.append("PREPROCESSING PIPELINE FINAL REPORT")
    report.append("=" * 60)
    report.append(f"Execution Time: {duration}")
    report.append("")
    
    report.append("RATINGS DATA SPLITS:")
    report.append(f"  - Train: {len(train_df):,} rows | {train_df['user_id'].nunique():,} users | {train_df['parent_asin'].nunique():,} items")
    report.append(f"  - Val:   {len(val_df):,} rows | {val_df['user_id'].nunique():,} users | {val_df['parent_asin'].nunique():,} items")
    report.append(f"  - Test:  {len(test_df):,} rows | {test_df['user_id'].nunique():,} users | {test_df['parent_asin'].nunique():,} items")
    report.append("")
    
    report.append("METADATA:")
    report.append(f"  - Cleaned Items: {len(meta_df):,}")
    report.append("")
    
    report.append("FEATURES ADDED:")
    report.append("  - helpful_vote_norm, confidence_norm, final_weight_norm")
    report.append("  - user_verified_ratio, item_avg_rating, is_weekend")
    report.append("=" * 60)
    
    report_text = "\n".join(report)
    
    report_path = Path("data/processed/preprocessing_report.txt")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report_text)
        
    logger.info("\n" + report_text)
    logger.info(f"Report saved to: {report_path}")

def main():
    start_time = datetime.now()
    logger.info("Starting Preprocessing Pipeline...")

    # ============================================
    # 1. LOAD CONFIG
    # ============================================
    config_path = Path("configs/data_config.yaml")
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    paths_cfg = config['paths']
    prep_cfg = config['preprocessing']
    chunk_size = config['dataset']['electronics']['chunk_size']
    
    processed_dir = Path(paths_cfg['processed_data'])
    processed_dir.mkdir(parents=True, exist_ok=True)
    intermediate_ratings_path = processed_dir / "intermediate_ratings.parquet"
    ratings_raw_path = Path(paths_cfg['raw_data']) / "Electronics_ratings.parquet"
    metadata_raw_path = Path(paths_cfg['raw_data']) / "Electronics_metadata.parquet"

    # ============================================
    # 2. PHASE 1: ROW-WISE CHUNKING (RATINGS)
    # ============================================
    logger.info(f"Phase 1: Processing large ratings file in chunks of {chunk_size:,}...")
    parquet_file = pq.ParquetFile(ratings_raw_path)
    writer = None
    
    current_chunk = 1

    for batch in parquet_file.iter_batches(batch_size=chunk_size):
        chunk = batch.to_pandas()
        logger.info(f"  Processing chunk {current_chunk}...")
        
        chunk = replace_none_strings(chunk, columns=prep_cfg['none_replacement']['columns'])
        chunk = remove_missing_values(chunk, subset=prep_cfg['missing_values']['subset'])
        chunk = remove_duplicates(chunk, subset=prep_cfg['deduplication']['subset'])
        chunk = drop_useless_columns(chunk, columns=prep_cfg['drop_columns'])
        chunk = convert_timestamp(chunk, column='timestamp')
        chunk = convert_to_integer(chunk, columns=prep_cfg['convert_to_integer']['columns'])
        
        chunk = handle_outliers(chunk, columns=prep_cfg['outliers']['columns'])
        chunk = filter_text(
            chunk, 
            column='text', 
            min_words=prep_cfg['text_filter']['min_words'], 
            max_words=prep_cfg['text_filter']['max_words']
        )
        
        table_chunk = pa.Table.from_pandas(chunk)
        if writer is None:
            writer = pq.ParquetWriter(intermediate_ratings_path, table_chunk.schema)
        writer.write_table(table_chunk)
        current_chunk += 1

    if writer: 
        writer.close()
    logger.info("Phase 1 Complete: Intermediate ratings saved.")
    gc.collect()

    # ============================================
    # 3. PHASE 2: GLOBAL OPERATIONS (RATINGS)
    # ============================================
    logger.info("Phase 2: Loading intermediate data for global operations...")
    
    columns_to_keep = [
        'rating', 'asin', 'parent_asin', 'user_id', 
        'timestamp', 'helpful_vote', 'verified_purchase'
    ]
    
    df = pd.read_parquet(
        intermediate_ratings_path,
        columns=columns_to_keep
    )
    
    logger.info("  Applying Optimized Spam Detection...")
    df = memory_optimized_detect_spam(
        df, 
        max_reviews_per_day=prep_cfg['spam_detection']['max_reviews_per_day'], 
        min_time_gap=prep_cfg['spam_detection']['min_time_gap']
    )
    
    logger.info("  Applying Deduplication...")
    df = deduplicate_user_item(
        df, 
        user_col=prep_cfg['implicit_feedback']['user_col'], 
        item_col=prep_cfg['implicit_feedback']['item_col']
    )
    
    logger.info("  Applying Iterative K-Core...")
    df = apply_iterative_k_core(
        df, 
        columns_config=prep_cfg['iterative_k_core']['columns_config'], 
        max_iterations=prep_cfg['iterative_k_core']['max_iterations']
    )
    
    logger.info("  Filtering Top N Items...")
    df = filter_top_n_items(
        df, 
        item_col=prep_cfg['filter_top_n']['item_col'], 
        top_n=prep_cfg['filter_top_n']['top_n']
    )

    logger.info("  Calculating Weights and Encoders...")
    df = apply_confidence_weight(
        df, 
        verified_col='verified_purchase', 
        weight_col='confidence_weight', 
        verified_weight=prep_cfg['review_weight']['verified'], 
        unverified_weight=prep_cfg['review_weight']['unverified']
    )
    
    df = set_implicit_feedback_weight(df, weight_col='implicit_weight', interaction_weight=1.0)
    df['final_weight'] = df['confidence_weight'] * df['implicit_weight']

    df['parent_asin_original'] = df['parent_asin'].copy()
    df, encoders = encode_labels(df, columns=prep_cfg['encode']['columns'], encoders=None)
    joblib.dump(encoders, processed_dir / "encoders.pkl")
    
    df = add_user_segment(df, light_max=2, medium_max=10)
    
    # ============================================
    # LATE BINDING: CHUNKED TEXT RECOVERY
    # ============================================
    logger.info("  Recovering text columns safely using chunked streaming...")
    valid_users = set(df['user_id'])
    text_chunks = []
    
    intermediate_pq = pq.ParquetFile(intermediate_ratings_path)
    for batch in intermediate_pq.iter_batches(batch_size=chunk_size, columns=['user_id', 'parent_asin', 'timestamp', 'text', 'title']):
        chunk = batch.to_pandas()
        chunk = chunk[chunk['user_id'].isin(valid_users)]
        if not chunk.empty:
            text_chunks.append(chunk)
            
    if text_chunks:
        df_text = pd.concat(text_chunks, ignore_index=True)
        df_text = df_text.drop_duplicates(subset=['user_id', 'parent_asin', 'timestamp'])
        df = pd.merge(df, df_text, on=['user_id', 'parent_asin', 'timestamp'], how='left')
        del df_text
        del text_chunks
        gc.collect()
        
    logger.info(f"Phase 2 Complete. Final ratings shape with text columns: {df.shape}")

    # ============================================
    # 4. PHASE 3: METADATA CHUNKED FILTERING
    # ============================================
    logger.info("Phase 3: Processing Metadata on-the-fly...")
    meta_parquet = pq.ParquetFile(metadata_raw_path)
    train_items = set(df['parent_asin_original'].unique())
    meta_chunks = []

    for batch in meta_parquet.iter_batches(batch_size=chunk_size):
        meta_chunk = batch.to_pandas()
        meta_chunk = meta_chunk[meta_chunk['parent_asin'].isin(train_items)]
        
        if not meta_chunk.empty:
            meta_chunk = replace_none_strings(meta_chunk, columns=prep_cfg['none_replacement']['columns'])
            meta_chunk = remove_missing_values(meta_chunk, subset=['parent_asin', 'description'])
            meta_chunk = remove_duplicates(meta_chunk, subset=['parent_asin'])
            meta_chunks.append(meta_chunk)

    df_meta = pd.concat(meta_chunks, ignore_index=True) if meta_chunks else pd.DataFrame()
    df_meta = add_store_tier(df_meta, small_max=2, large_min=50, rating_threshold=4.4)
    df_meta.to_parquet(processed_dir / paths_cfg['metadata_file'], index=False)
    logger.info(f"Phase 3 Complete. Metadata saved with shape: {df_meta.shape}")

    # ============================================
    # 5. PHASE 4: TIME-BASED SPLIT & SCALING
    # ============================================
    logger.info("Phase 4: Data Splitting, Scaling, and Feature Engineering...")
    split_cfg = prep_cfg['split']
    train_df, val_df, test_df = time_based_split(
        df, 
        column='timestamp', 
        val_year=split_cfg['val_year'], 
        test_year=split_cfg['test_year']
    )

    logger.info("  Sanitizing numerical columns before scaling...")
    scale_cols = ['helpful_vote', 'confidence_weight', 'final_weight']
    for d in [train_df, val_df, test_df]:
        for col in scale_cols:
            d[col] = d[col].replace([np.inf, -np.inf], np.nan).fillna(0)

    scaler_helpful = MinMaxScaler()
    scaler_conf = MinMaxScaler()
    scaler_final = MinMaxScaler()

    train_df['helpful_vote_norm'] = scaler_helpful.fit_transform(train_df[['helpful_vote']])
    val_df['helpful_vote_norm'] = scaler_helpful.transform(val_df[['helpful_vote']])
    test_df['helpful_vote_norm'] = scaler_helpful.transform(test_df[['helpful_vote']])

    train_df['confidence_norm'] = scaler_conf.fit_transform(train_df[['confidence_weight']])
    val_df['confidence_norm'] = scaler_conf.transform(val_df[['confidence_weight']])
    test_df['confidence_norm'] = scaler_conf.transform(test_df[['confidence_weight']])

    train_df['final_weight_norm'] = scaler_final.fit_transform(train_df[['final_weight']])
    val_df['final_weight_norm'] = scaler_final.transform(val_df[['final_weight']])
    test_df['final_weight_norm'] = scaler_final.transform(test_df[['final_weight']])

    joblib.dump(
        {'helpful': scaler_helpful, 'confidence': scaler_conf, 'final_weight': scaler_final}, 
        processed_dir / "scalers.pkl"
    )

    train_df = add_features(train_df)
    
    user_verified_map = train_df.groupby('user_id')['verified_purchase'].mean().to_dict()
    item_avg_rating_map = train_df.groupby('parent_asin')['rating'].mean().to_dict()
    train_rating_mean = train_df['rating'].mean()

    for d in [val_df, test_df]:
        d['user_verified_ratio'] = d['user_id'].map(user_verified_map).fillna(0.5)
        d['item_avg_rating'] = d['parent_asin'].map(item_avg_rating_map).fillna(train_rating_mean)
        d['is_weekend'] = d['timestamp'].dt.dayofweek.isin([5, 6]).astype(int)

    # ============================================
    # 6. SAVE FINAL FILES
    # ============================================
    logger.info("Saving final datasets...")
    train_df.to_parquet(processed_dir / paths_cfg['train_file'], index=False)
    val_df.to_parquet(processed_dir / paths_cfg['val_file'], index=False)
    test_df.to_parquet(processed_dir / paths_cfg['test_file'], index=False)

    generate_report(train_df, val_df, test_df, df_meta, start_time)
    logger.info("Pipeline executed successfully.")

if __name__ == "__main__":
    main()