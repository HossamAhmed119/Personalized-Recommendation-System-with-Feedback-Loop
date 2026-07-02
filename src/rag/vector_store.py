import os
import re
import html
import yaml
import pandas as pd
import numpy as np
import torch
from tqdm import tqdm
import chromadb
from chromadb.utils import embedding_functions

class ComprehensiveRAGBuilder:
    def __init__(self, raw_meta_path, db_persist_dir):
        """
        Initialize database paths and ChromaDB client for Parquet processing
        with GPU acceleration.
        """
        self.raw_meta_path = raw_meta_path
        self.db_persist_dir = db_persist_dir
        
        os.makedirs(self.db_persist_dir, exist_ok=True)
        self.chroma_client = chromadb.PersistentClient(path=self.db_persist_dir)
        
        # Detect and assign the processing device (GPU if available)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"[INFO] Initializing Embedding Model on device: {self.device.upper()}")
        
        # Pass the device explicitly to force GPU usage
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        
        self.collection = self.chroma_client.get_or_create_collection(
            name="amazon_full_catalog",
            embedding_function=self.embedding_func
        )

    def _clean_text(self, text):
        """
        Fast text cleaning removing HTML tags and normalizing whitespaces
        """
        if text is None:
            return ""
        if isinstance(text, (float, int)) and pd.isna(text):
            return ""
        if not isinstance(text, str):
            text = str(text)
            
        clean_re = re.compile('<.*?>')
        text = re.sub(clean_re, ' ', text)
        text = html.unescape(text)
        return " ".join(text.split())

    def _process_list_column(self, val):
        """
        Safely convert lists/arrays in dataframe cells to strings,
        avoiding the ambiguous truth value error of numpy arrays.
        """
        if val is None:
            return ""
            
        if isinstance(val, np.ndarray):
            if val.size == 0:
                return ""
            return " ".join([str(v) for v in val if v is not None and str(v) != 'nan'])
            
        if isinstance(val, list):
            if len(val) == 0:
                return ""
            return " ".join([str(v) for v in val if v is not None and str(v) != 'nan'])
            
        if pd.isna(val):
            return ""
            
        return str(val)

    def build_database(self, batch_size=2000):
        """
        Read the Parquet metadata file, process, and upsert chunks in batches
        """
        print(f"[INFO] Loading Parquet metadata from: {self.raw_meta_path}")
        
        if not os.path.exists(self.raw_meta_path):
            print(f"[ERROR] Metadata file not found at {self.raw_meta_path}")
            return

        try:
            columns_to_read = ['parent_asin', 'title', 'description', 'features', 'categories']
            df = pd.read_parquet(self.raw_meta_path, columns=columns_to_read)
            df = df.dropna(subset=['parent_asin', 'title'])
            print(f"[INFO] Total valid products loaded into memory: {len(df)}")
            
        except Exception as e:
            print(f"[ERROR] Failed to load Parquet file: {e}")
            return

        documents = []
        metadatas = []
        ids = []
        
        print("[INFO] Processing metadata and building RAG chunks...")
        
        records = df.to_dict('records')
        
        for record in tqdm(records):
            asin = str(record['parent_asin'])
            title = self._clean_text(record['title'])
            
            if not title or title.lower() == 'nan':
                continue
                
            desc_text = self._process_list_column(record.get('description'))
            feat_text = self._process_list_column(record.get('features'))
            full_desc = self._clean_text(desc_text + " " + feat_text)
            
            if len(full_desc) < 10:
                continue
                
            cat_text = self._process_list_column(record.get('categories'))
            category = cat_text.replace(' ', ' > ') if cat_text else "Unknown Category"
            
            structured_chunk = (
                f"Product ID: {asin}\n"
                f"Title: {title}\n"
                f"Category: {category}\n"
                f"Description and Features: {full_desc}"
            )
            
            documents.append(structured_chunk)
            metadatas.append({
                "parent_asin": asin,
                "title": title[:100]
            })
            ids.append(asin)
            
            if len(documents) >= batch_size:
                self.collection.upsert(
                    documents=documents,
                    metadatas=metadatas,
                    ids=ids
                )
                documents, metadatas, ids = [], [], []

        if documents:
            self.collection.upsert(
                documents=documents,
                metadatas=metadatas,
                ids=ids
            )
            
        print("\n[SUCCESS] Full Catalog Vector Database built successfully!")
        print(f"[INFO] Location: {self.db_persist_dir}")
        print(f"[INFO] Total products embedded and ready for RAG: {self.collection.count()}")

def load_config(config_path="configs/data_config.yaml"):
    """
    Load YAML configuration file
    """
    with open(config_path, "r", encoding="utf-8") as file:
        return yaml.safe_load(file)

if __name__ == "__main__":
    config = load_config()
    
    raw_meta_path = config['paths']['raw_meta_parquet']
    db_persist_dir = config['paths']['chroma_db_dir']
    
    print(f"[INFO] Starting RAG Database Build Process...")
    
    builder = ComprehensiveRAGBuilder(raw_meta_path, db_persist_dir)
    builder.build_database()