import requests
import gzip
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
import pyarrow as pa
import pyarrow.parquet as pq
from src.utils.config_loader import load_config

# Important metadata columns
METADATA_COLUMNS = [
    'parent_asin',
    'title',
    'description', 
    'features',
    'categories',
    'price',
    'store',
    'average_rating',
    'rating_number'
]

def download_file(url: str, save_path: str) -> None:
    """Downloads a file with progress bar"""
    
    if Path(save_path).exists():
        print(f"[INFO] File already exists: {save_path}")
        return
    
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    
    response = requests.get(url, stream=True)
    total_size = int(response.headers.get("content-length", 0))
    
    with open(save_path, "wb") as f, tqdm(
        total=total_size,
        unit="B",
        unit_scale=True,
        desc=Path(save_path).name
    ) as bar:
        for chunk in response.iter_content(chunk_size=8192):
            f.write(chunk)
            bar.update(len(chunk))

    print(f"[INFO] Downloaded: {save_path}")


def extract_jsonl_gz(gz_path: str, output_path: str,
                     sample_size: int, chunk_size: int,
                     columns: list = None) -> None:
    """
    Extracts .jsonl.gz file and saves as parquet in chunks.
    """
    if Path(output_path).exists():
        print(f"[INFO] File already exists: {output_path}")
        return

    print(f"[INFO] Extracting: {gz_path}")
    
    records = []
    writer = None
    total_rows = 0

    with gzip.open(gz_path, "rb") as f:
        for i, line in enumerate(tqdm(f, total=sample_size)):
            if i >= sample_size:
                break
                
            record = json.loads(line)
            
            # خد الـ columns المهمة بس
            if columns:
                record = {k: record.get(k) for k in columns}
            
            records.append(record)
            
            if len(records) >= chunk_size:
                df_chunk = pd.DataFrame(records)
                
                # clean price column
                if 'price' in df_chunk.columns:
                    df_chunk['price'] = pd.to_numeric(
                        df_chunk['price'], errors='coerce'
                    )
                
                table = pa.Table.from_pandas(df_chunk, safe=False)
                
                if writer is None:
                    writer = pq.ParquetWriter(output_path, table.schema)
                
                writer.write_table(table)
                total_rows += len(records)
                records = []
                print(f"[INFO] Written {total_rows} rows...")
    
    # save remaining records
    if records:
        df_chunk = pd.DataFrame(records)
        
        if 'price' in df_chunk.columns:
            df_chunk['price'] = pd.to_numeric(
                df_chunk['price'], errors='coerce'
            )
        
        table = pa.Table.from_pandas(df_chunk, safe=False)
        if writer is None:
            writer = pq.ParquetWriter(output_path, table.schema)
        writer.write_table(table)
        total_rows += len(records)
    
    if writer:
        writer.close()
    
    print(f"[INFO] Saved: {output_path} | Total rows: {total_rows}")


def main():
    config = load_config("configs/data_config.yaml")
    
    ratings_url  = config['dataset']['electronics']['ratings_url']
    metadata_url = config['dataset']['electronics']['metadata_url']
    sample_size  = config['dataset']['electronics']['sample_size']
    chunk_size   = config['dataset']['electronics']['chunk_size']
    raw_path     = config['paths']['raw_data']

    # ---- Ratings ----
    gz_path      = raw_path + "Electronics_ratings.jsonl.gz"
    parquet_path = raw_path + "Electronics_ratings.parquet"
    
    download_file(ratings_url, gz_path)
    extract_jsonl_gz(gz_path, parquet_path, sample_size, chunk_size)

    # ---- Metadata ----
    gz_path_meta      = raw_path + "Electronics_metadata.jsonl.gz"
    parquet_path_meta = raw_path + "Electronics_metadata.parquet"

    download_file(metadata_url, gz_path_meta)
    extract_jsonl_gz(
    gz_path_meta, 
    parquet_path_meta, 
    sample_size, 
    chunk_size,
    columns = METADATA_COLUMNS  
    )
if __name__ == "__main__":
    main()