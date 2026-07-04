import logging
import os
from typing import Dict, List, Optional

import chromadb
import pandas as pd
import requests
import yaml
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class ProductRetriever:
    def __init__(self, app_config_path: str, data_config_path: str):
        with open(app_config_path, "r", encoding="utf-8") as f:
            self.app_config = yaml.safe_load(f)
        with open(data_config_path, "r", encoding="utf-8") as f:
            self.data_config = yaml.safe_load(f)

        self.item_id_col = self.app_config["schema"]["item_id_col"]
        self.title_col = self.app_config["schema"].get("title_col", "title")
        self.description_col = self.app_config["schema"].get("description_col", "description")

        self._chroma_client: Optional[chromadb.ClientAPI] = None
        self._collection = None
        self.products_df: Optional[pd.DataFrame] = None

        api_key_env_var = self.app_config["rag"]["api_key_env_var"]
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            logger.warning(
                "Environment variable '%s' is not set. RAG calls will fail until it is configured.",
                api_key_env_var,
            )

    def load_products(self, products_df: pd.DataFrame) -> None:
        self.products_df = products_df

    def _get_client(self):
        if self._chroma_client is None:
            persist_dir = self.app_config["chromadb"]["persist_directory"]
            self._chroma_client = chromadb.PersistentClient(path=persist_dir)
            logger.info("Initialized ChromaDB client at %s", persist_dir)
        return self._chroma_client

    def _get_collection(self):
        if self._collection is None:
            client = self._get_client()
            collection_name = self.app_config["chromadb"]["collection_name"]
            embed_fn = embedding_functions.DefaultEmbeddingFunction()
            self._collection = client.get_or_create_collection(
                name=collection_name, embedding_function=embed_fn
            )
            logger.info("Connected to ChromaDB collection '%s'", collection_name)
        return self._collection

    def index_products(self, batch_size: int = 256) -> None:
        if self.products_df is None:
            raise RuntimeError("Call load_products() before index_products().")

        collection = self._get_collection()

        if collection.count() >= len(self.products_df):
            logger.info("ChromaDB collection already indexed (%d items). Skipping.", collection.count())
            return

        docs, ids, metadatas = [], [], []
        for _, row in self.products_df.iterrows():
            title_val = row.get(self.title_col, "")
            if isinstance(title_val, pd.Series):
                title_val = title_val.iloc[0]
            
            text = str(title_val)

            if self.description_col and self.description_col in row:
                desc_val = row[self.description_col]
                if isinstance(desc_val, pd.Series):
                    desc_val = desc_val.iloc[0]
                
                desc_str = str(desc_val).strip()
                if desc_str and desc_str.lower() not in ('nan', 'none', '<na>', 'nat'):
                    text += " " + desc_str

            item_id = row[self.item_id_col]
            if isinstance(item_id, pd.Series):
                item_id = item_id.iloc[0]

            docs.append(text)
            ids.append(str(item_id))
            metadatas.append({"parent_asin": str(item_id)})

        try:
            for start in range(0, len(docs), batch_size):
                end = start + batch_size
                collection.upsert(
                    documents=docs[start:end],
                    ids=ids[start:end],
                    metadatas=metadatas[start:end],
                )
            logger.info("Indexed %d products into ChromaDB.", len(docs))
        except Exception:
            logger.exception("Failed to index products into ChromaDB.")
            raise

    def semantic_search(self, query: str, top_k: int = 5) -> List[Dict]:
        collection = self._get_collection()
        try:
            results = collection.query(query_texts=[query], n_results=top_k)
        except Exception:
            logger.exception("ChromaDB query failed for query='%s'", query)
            return []

        hits = []
        documents = results.get("documents", [[]])[0]
        metadatas = results.get("metadatas", [[]])[0]
        distances = results.get("distances", [[]])[0]

        for doc, meta, dist in zip(documents, metadatas, distances):
            hits.append({"parent_asin": meta["parent_asin"], "document": doc, "distance": dist})
        return hits

    @staticmethod
    def _build_context(hits: List[Dict]) -> str:
        lines = [f"- parent_asin: {hit['parent_asin']} | {hit['document'][:200]}" for hit in hits]
        return "\n".join(lines)

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        wants_recommendations: bool = False,
    ) -> str:
        if not self.api_key:
            return "The assistant is not fully configured yet (missing OpenRouter API key)."

        messages = [{"role": "system", "content": self.app_config["rag"]["system_prompt"]}]

        if conversation_history:
            messages.extend(conversation_history)

        if wants_recommendations:
            hits = self.semantic_search(user_message, top_k=5)
            context = self._build_context(hits)
            augmented_message = (
                f"{user_message}\n\nRelevant catalog items "
                f"(reference them by parent_asin only, do not invent others):\n{context}"
            )
            messages.append({"role": "user", "content": augmented_message})
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            response = requests.post(
                url=f"{self.app_config['rag']['api_base']}/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": self.app_config["rag"]["model"],
                    "messages": messages,
                    "temperature": self.app_config["rag"]["temperature"],
                    "max_tokens": self.app_config["rag"]["max_tokens"],
                },
                timeout=30,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except Exception:
            logger.exception("OpenRouter chat completion failed for message='%s'", user_message)
            return "Sorry, I ran into an issue reaching the assistant. Please try again."