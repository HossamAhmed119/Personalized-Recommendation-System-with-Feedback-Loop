"""
Extract images mapping for the top 10,000 filtered products.
Reads line-by-line directly from the raw compressed jsonl.gz file.
"""

import gzip
import json
import pandas as pd
from pathlib import Path

from src.utils.config_loader import load_data_config, load_app_config

def extract_product_images():
    print("Loading configurations...")
    
    project_root = Path(__file__).resolve().parents[2]
    data_config_path = project_root / "configs" / "data_config.yaml"
    app_config_path = project_root / "configs" / "app_config.yaml"
    
    data_config = load_data_config(str(data_config_path))
    app_config = load_app_config(str(app_config_path))
    
    raw_dir = project_root / data_config['paths']['raw_data']
    raw_gz_path = raw_dir / "Electronics_metadata.jsonl.gz"
    
    processed_meta_path = project_root / app_config['paths']['processed_items']
    output_json = project_root / app_config['database']['image_mapping']
    
    if not raw_gz_path.exists():
        print(f"Error: Raw compressed file not found at {raw_gz_path}")
        return

    print("Loading target ASINs from processed data...")
    processed_df = pd.read_parquet(processed_meta_path, columns=['parent_asin'])
    target_asins = set(processed_df['parent_asin'].unique())
    
    image_mapping = {}
    success_count = 0
    
    print(f"Scanning raw gzip file for {len(target_asins)} products...")
    
    with gzip.open(raw_gz_path, 'rt', encoding='utf-8') as f:
        for line in f:
            try:
                record = json.loads(line)
                asin = record.get('parent_asin')
                
                if asin in target_asins:
                    images_data = record.get('images', [])
                    found_image = False
                    
                    if isinstance(images_data, list) and len(images_data) > 0:
                        first_img = images_data[0]
                        
                        # Corrected: Assigned the string directly without [0] indexing
                        if first_img.get('hi_res'):
                            image_mapping[asin] = first_img['hi_res']
                            found_image = True
                        elif first_img.get('large'):
                            image_mapping[asin] = first_img['large']
                            found_image = True
                    
                    if found_image:
                        success_count += 1
                        
                    if len(image_mapping) >= len(target_asins):
                        print("All target images found. Stopping search early.")
                        break
                        
            except json.JSONDecodeError:
                continue

    fallback_count = len(target_asins) - len(image_mapping)
    
    output_json.parent.mkdir(parents=True, exist_ok=True)
    with open(output_json, 'w', encoding='utf-8') as f:
        json.dump(image_mapping, f, indent=4)
        
    print("\n--- Extraction Summary ---")
    print(f"Total Target Products: {len(target_asins)}")
    print(f"Successfully Extracted Images: {success_count}")
    print(f"Failed/Fallback to Placeholder: {fallback_count}")
    print(f"Mapping saved to: {output_json}")

if __name__ == "__main__":
    extract_product_images()