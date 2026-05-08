import requests
import gzip
import json
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from src.utils.config_loader import load_config

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


def extract_jsonl_gz(gz_path: str, csv_path: str, sample_size: int) -> None:
    """Extracts .jsonl.gz file and saves as CSV"""

    if Path(csv_path).exists():
        print(f"[INFO] CSV already exists: {csv_path}")
        return

    print(f"[INFO] Extracting: {gz_path}")
    records = []

    with gzip.open(gz_path, "rb") as f:
        for i, line in enumerate(tqdm(f, total=sample_size)):
            if i >= sample_size:
                break
            records.append(json.loads(line))

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"[INFO] Saved CSV: {csv_path} | Rows: {len(df)}")

def main():
    config = load_config("configs/data_config.yaml")
    
    ratings_url = config['dataset']['electronics']['ratings_url']
    metadata_url = config['dataset']['electronics']['metadata_url']
    sample_size = config['dataset']['electronics']['sample_size']
    raw_path    = config['paths']['raw_data']

    # ---- Ratings ----
    gz_path  = raw_path + "Electronics_ratings.jsonl.gz"
    csv_path = raw_path + "Electronics_ratings.csv"
    
    download_file(ratings_url, gz_path)
    extract_jsonl_gz(gz_path, csv_path, sample_size)
   
    # ---- Metadata ----
    gz_path_meta  = raw_path + "Electronics_metadata.jsonl.gz"
    csv_path_meta = raw_path + "Electronics_metadata.csv"

    download_file(metadata_url, gz_path_meta)
    extract_jsonl_gz(gz_path_meta, csv_path_meta, sample_size)



if __name__ == "__main__":
    main()