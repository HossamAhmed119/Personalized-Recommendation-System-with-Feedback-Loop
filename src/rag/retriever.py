import logging
import os
from typing import Dict, List, Optional
import requests
import yaml
import chromadb
from chromadb.utils import embedding_functions

logger = logging.getLogger(__name__)

class ProductRetriever:
    def __init__(self, app_config_path: str, data_config_path: str):
        with open(app_config_path, "r", encoding="utf-8") as f:
            self.app_config = yaml.safe_load(f)
        with open(data_config_path, "r", encoding="utf-8") as f:
            self.data_config = yaml.safe_load(f)

        api_key_env_var = self.app_config.get("rag", {}).get("api_key_env_var", "OPENROUTER_API_KEY")
        self.api_key = os.environ.get(api_key_env_var)
        if not self.api_key:
            logger.warning(f"Environment variable '{api_key_env_var}' is not set. RAG calls will fail.")

        db_persist_dir = self.app_config['chromadb']['persist_directory']
        collection_name = self.app_config['chromadb']['collection_name']
        
        self.chroma_client = chromadb.PersistentClient(path=db_persist_dir)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2"
        )
        
        try:
            self.collection = self.chroma_client.get_collection(
                name=collection_name,
                embedding_function=self.embedding_func
            )
            logger.info(f"Successfully connected to ChromaDB: {collection_name}")
        except Exception as e:
            logger.error(f"Failed to load ChromaDB collection: {e}")
            self.collection = None

    def search_products(self, query: str, top_k: int = 5) -> str:
        if not self.collection:
            return ""
            
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results.get('documents', [[]])[0]
        if not documents:
            return ""
            
        return "\n\n--- PRODUCT ---\n".join(documents)

    def chat(
        self,
        user_message: str,
        conversation_history: Optional[List[Dict]] = None,
        wants_recommendations: bool = False,
    ) -> str:
        if not self.api_key:
            return "The assistant is not fully configured yet (missing OpenRouter API key)."

        system_prompt = (
            "You are an expert Amazon product recommendation assistant.\n\n"
            "Strict Instructions:\n"
            "1. GREETINGS: If the user's message is a simple greeting (e.g., 'hi', 'hello'), simply say hello politely and ask how you can help. Do NOT create tables.\n"
            "2. RELEVANCE CHECK: For product queries, first evaluate if the products provided in the context block are ACTUALLY relevant to the user's request.\n"
            "3. IRRELEVANT CONTEXT: IF NO RELEVANT PRODUCTS ARE FOUND in the context (e.g., the user asks for a laptop but context only has flash drives), politely apologize and state that you do not have those items in the current catalog. DO NOT create a table or list irrelevant products.\n"
            "4. RELEVANT CONTEXT: IF RELEVANT PRODUCTS ARE FOUND, you MUST analyze them, rank them from best to lowest match, and provide a clear breakdown using a comprehensive comparison table (e.g., Performance, Value for money).\n"
            "5. NO HALLUCINATION: Rely ONLY on the provided context. Do not invent specifications or products."
        )

        messages = [{"role": "system", "content": system_prompt}]

        if conversation_history:
            messages.extend(conversation_history[-4:])

        if wants_recommendations:
            context_block = self.search_products(user_message, top_k=5)
            
            if context_block:
                augmented_message = f"User Request: {user_message}\n\nAvailable Products Context:\n{context_block}"
            else:
                augmented_message = f"User Request: {user_message}\n\n(No specific products found in the catalog matching this request.)"
            
            messages.append({"role": "user", "content": augmented_message})
        else:
            messages.append({"role": "user", "content": user_message})

        try:
            model_id = self.app_config.get("rag", {}).get("model", "openai/gpt-oss-20b:free")
            
            response = requests.post(
                url=self.app_config["rag"]["api_base"] + "/chat/completions",
                headers={
                    "Authorization": f"Bearer {self.api_key}",
                    "Content-Type": "application/json",
                },
                json={
                    "model": model_id,
                    "messages": messages,
                    "temperature": 0.3,
                    "max_tokens": 2500,
                },
                timeout=45,
            )
            response.raise_for_status()
            payload = response.json()
            return payload["choices"][0]["message"]["content"]
        except Exception as e:
            logger.exception("LLM generation failed")
            return f"Sorry, I ran into an issue generating the response. Details: {str(e)}"