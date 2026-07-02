import os
import yaml
import torch
import chromadb
from chromadb.utils import embedding_functions
from openai import OpenAI

class ProductRetriever:
    def __init__(self, config_path="configs/data_config.yaml", llm_model="openai/gpt-oss-20b:free"):
        """
        Initialize the Retriever with ChromaDB for semantic search 
        and OpenRouter for LLM generation.
        """
        print("[INFO] Initializing Product Retriever...")
        
        # 1. Load Configuration
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Configuration file not found at {config_path}")
            
        with open(config_path, "r", encoding="utf-8") as file:
            config = yaml.safe_load(file)
            
        db_persist_dir = config['paths']['chroma_db_dir']
        
        # 2. Initialize Vector Database (ChromaDB)
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.chroma_client = chromadb.PersistentClient(path=db_persist_dir)
        self.embedding_func = embedding_functions.SentenceTransformerEmbeddingFunction(
            model_name="all-MiniLM-L6-v2",
            device=self.device
        )
        
        # Fetch the pre-built collection
        try:
            self.collection = self.chroma_client.get_collection(
                name="amazon_full_catalog",
                embedding_function=self.embedding_func
            )
            print(f"[INFO] Successfully connected to ChromaDB on device: {self.device.upper()}")
        except ValueError:
            raise ValueError("[ERROR] Collection 'amazon_full_catalog' not found. Please run vector_store.py first.")

        # 3. Initialize OpenRouter LLM Client
        self.api_key = os.getenv("OPENROUTER_API_KEY")
        if not self.api_key:
            print("[WARNING] OPENROUTER_API_KEY environment variable is not set. LLM features will fail.")
            
        self.llm_client = OpenAI(
            base_url="https://openrouter.ai/api/v1",
            api_key=self.api_key or "DUMMY_KEY"
        )
        # Note: Replace with your specific OpenRouter 20B model ID if needed
        self.model_id = llm_model 

    def search_products(self, query, top_k=5):
        """
        Search the vector database for products matching the user query.
        """
        print(f"\n[INFO] Searching for: '{query}'")
        results = self.collection.query(
            query_texts=[query],
            n_results=top_k
        )
        
        documents = results.get('documents', [[]])[0]
        metadatas = results.get('metadatas', [[]])[0]
        
        if not documents:
            return "No matching products found.", []
            
        # Format the retrieved documents into a clean context block
        context_block = "\n\n--- PRODUCT ---\n".join(documents)
        return context_block, metadatas

    def generate_recommendation(self, user_query, context_block):
        """
        Send the query and retrieved context to the LLM via OpenRouter.
        """
        # This  prompt forces the LLM to compare, rank, and list all retrieved top_k products
        system_prompt = (
            "You are an expert Amazon product recommendation assistant.\n"
            "Your task is to analyze, compare, and rank ALL the products provided in the context based on how well they match the user's request.\n\n"
            "Strict Instructions:\n"
            "1. You MUST list and evaluate ALL products present in the provided context block. Do not omit any product.\n"
            "2. Rank the products from best match to lowest match based on the user's criteria.\n"
            "3. For each product, provide a clear breakdown of why it is better or worse than the other options (e.g., Performance, RAM, Value for money).\n"
            "4. Format your response cleanly using Markdown headings, bullet points, and a comprehensive comparison table comparing all products side-by-side.\n"
            "5. Rely ONLY on the provided context. Do not hallucinate or invent specifications."
        )
        
        user_prompt = f"User Request: {user_query}\n\nAvailable Products Context:\n{context_block}"
        
        print(f"[INFO] Generating response using model: {self.model_id}...")
        
        try:
            response = self.llm_client.chat.completions.create(
                model=self.model_id,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.3,
                max_tokens=2500 # Increased tokens to fit the comprehensive 5-product comparison
            )
            return response.choices[0].message.content
            
        except Exception as e:
            return f"[ERROR] LLM Generation failed: {str(e)}"

    def ask(self, query, top_k=5):
        """
        End-to-end pipeline: Retrieve -> Generate -> Return
        """
        context_block, metadatas = self.search_products(query, top_k)
        
        if not metadatas:
            return "Sorry, I couldn't find any products matching your criteria in the catalog."
            
        llm_response = self.generate_recommendation(query, context_block)
        return llm_response

if __name__ == "__main__":
    # Test the Retriever Pipeline
    
    # IMPORTANT: Set your OpenRouter API Key for testing
    # Uncomment and replace the placeholder below, or set it in your terminal environment
    # os.environ["OPENROUTER_API_KEY"] = "sk-or-v1-..."
    
    # You can change the model ID here to your preferred 20B OpenRouter model
    CHOSEN_MODEL = "mistralai/mixtral-8x7b-instruct" 
    
    try:
        retriever = ProductRetriever(
            config_path="configs/data_config.yaml",
            llm_model=CHOSEN_MODEL
        )
        
        test_query = "I am looking for a 27-inch curved gaming monitor with at least 144Hz refresh rate."
        
        print("\n" + "="*50)
        print("TESTING RETRIEVAL AND GENERATION")
        print("="*50)
        
        final_answer = retriever.ask(test_query, top_k=3)
        
        print("\n[FINAL LLM RESPONSE]:\n")
        print(final_answer)
        
    except Exception as e:
        print(f"\n[CRITICAL ERROR] {e}")