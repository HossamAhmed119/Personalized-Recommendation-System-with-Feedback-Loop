from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Optional
import uvicorn
import os
from dotenv import load_dotenv

# Import the clean Retriever class from your RAG module
from src.rag.retriever import ProductRetriever

# Load environment variables from .env
load_dotenv()

# Initialize FastAPI application
app = FastAPI(
    title="Amazon Recommendation Engine API",
    description="API for the Hybrid Recommendation and RAG System",
    version="1.0.0"
)

# Define the structured request layout with oss20b as default
class ChatRequest(BaseModel):
    query: str
    top_k: Optional[int] = 5
    model_id: Optional[str] = "openai/gpt-oss-20b:free"

# Global placeholder to store the initialized retriever instance
rag_system = None

@app.on_event("startup")
def startup_event():
    """
    Initialize the heavy RAG system components once during server startup.
    """
    global rag_system
    print("[INFO] Starting up API Server and initializing RAG components...")
    
    if not os.getenv("OPENROUTER_API_KEY"):
        print("[WARNING] OPENROUTER_API_KEY is missing from environment. LLM calls will fail.")
        
    try:
        # Core initialization using configuration path and default model
        rag_system = ProductRetriever(
            config_path="configs/data_config.yaml",
            llm_model="oss20b" 
        )
        print("[SUCCESS] RAG system components loaded into memory successfully.")
    except Exception as e:
        print(f"[ERROR] Failed to initialize RAG system during startup: {e}")

@app.post("/api/recommend")
def get_recommendation(request: ChatRequest):
    """
    Production endpoint to receive user natural language queries, 
    perform vector search, and return LLM generated responses.
    """
    if rag_system is None:
        raise HTTPException(
            status_code=500, 
            detail="The RAG system pipeline is unavailable or failed to initialize."
        )
        
    try:
        # Dynamically switch the model identifier if overridden in the request
        rag_system.model_id = request.model_id
        
        # Execute the end-to-end RAG pipeline
        response = rag_system.ask(query=request.query, top_k=request.top_k)
        
        return {
            "status": "success",
            "query": request.query,
            "used_model": request.model_id,
            "response": response
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    # Start the local development server on port 8000
    print("[INFO] Launching local Uvicorn deployment server...")
    uvicorn.run("src.api.main:app", host="0.0.0.0", port=8000, reload=True)