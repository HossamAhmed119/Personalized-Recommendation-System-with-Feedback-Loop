# Personalized Recommendation System with Feedback Loop

## Project Overview

A comprehensive recommendation system that leverages multiple machine learning approaches including Collaborative Filtering, Deep Learning, and Large Language Models (LLM) with a Retrieval-Augmented Generation (RAG) pipeline for intelligent product recommendations.

## Team Information

- **Team**: Group 2
- **Supervisor**: George Samuel

## Project Structure

```
recommendation-system/
├── configs/                 # Configuration files
│   ├── app_config.yaml
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── experiments/         # Experiment configurations
│       ├── exp_001.yaml
│       └── exp_002.yaml
├── data/                    # Data folder
│   ├── raw/                 # Raw datasets
│   ├── processed/           # Processed datasets
│   └── embeddings/          # Pre-computed embeddings
├── src/                     # Source code
│   ├── api/                 # FastAPI endpoints
│   ├── data_pipeline/       # Data processing
│   │   ├── ingest.py
│   │   ├── preprocess.py
│   │   └── features.py
│   ├── models/              # ML Models
│   │   ├── cf_model.py      # Collaborative Filtering
│   │   ├── deep_model.py    # Deep Learning Model
│   │   └── llm_reranker.py  # LLM-based Reranker
│   ├── rag/                 # RAG Pipeline
│   │   ├── embedder.py
│   │   ├── vector_store.py
│   │   └── retriever.py
│   ├── services/            # Business logic
│   │   └── recommender_service.py
│   └── utils/               # Utilities
│       ├── config_loader.py
│       ├── logger.py
│       └── helpers.py
├── mlops/                   # MLOps & Training
│   ├── train.py
│   ├── evaluate.py
│   └── mlflow_tracking.py
├── tests/                   # Unit tests
├── docs/                    # Documentation
├── requirements.txt         # Python dependencies
├── Dockerfile              # Docker configuration
├── docker-compose.yml      # Docker Compose
└── setup_project.py        # Project setup script
```

## Key Features

### 1. **Data Pipeline**
- CSV and JSONL data ingestion
- Preprocessing and feature engineering
- Embeddings generation and storage

### 2. **Recommendation Models**
- **Collaborative Filtering**: Based on user-item interactions
- **Deep Learning**: Neural network-based recommendations
- **LLM Reranker**: GPT-based intelligent re-ranking

### 3. **RAG Pipeline**
- Vector store management (FAISS)
- Sentence transformers for embeddings
- Retriever for context-aware recommendations

### 4. **API Services**
- FastAPI-based REST API
- Async/await support
- Pydantic validation

### 5. **MLOps**
- MLflow tracking and experiment management
- Data validation with Great Expectations
- Model evaluation framework

## Technology Stack

### Core Libraries
- **Data**: pandas, numpy, scikit-learn
- **Models**: torch, transformers, scipy, implicit
- **RAG & LLM**: langchain, langchain-openai, FAISS, sentence-transformers
- **API**: FastAPI, uvicorn
- **MLOps**: MLflow, Great Expectations
- **Cloud**: Azure AI/ML

### Tools & Utilities
- python-dotenv, pyyaml, loguru
- pytest for testing
- Docker for containerization

## Installation

### 1. Clone Repository
```bash
git clone <repository-url>
cd recommendation-system
```

### 2. Setup Virtual Environment
```bash
python -m venv system_env
system_env\Scripts\activate  # Windows
# source system_env/bin/activate  # Linux/Mac
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Configuration
```bash
python setup_project.py
```

## Configuration

Configuration files are located in the `configs/` directory:

- **app_config.yaml**: Application settings
- **data_config.yaml**: Data source and paths
- **model_config.yaml**: Model hyperparameters
- **training_config.yaml**: Training settings
- **experiments/**: Experiment-specific configurations

## Usage

### Running the API Server
```bash
uvicorn src.api.main:app --reload
```

### Training Models
```bash
python mlops/train.py --config configs/training_config.yaml
```

### Model Evaluation
```bash
python mlops/evaluate.py
```

### MLflow Tracking
```bash
mlflow ui
```

## Data

The project uses Electronics dataset:
- **Electronics_ratings.csv/jsonl.gz**: User ratings data
- **Electronics_metadata.csv/jsonl.gz**: Product metadata

Location: `data/raw/`

## Development Status

### Completed
- ✅ Project structure and scaffolding
- ✅ Requirements and dependencies
- ✅ Configuration system
- ✅ Data pipeline (ingest, preprocess, features)
- ✅ Model implementations (CF, Deep Learning, LLM Reranker)
- ✅ RAG pipeline (embedder, vector store, retriever)
- ✅ FastAPI application setup
- ✅ MLOps framework (MLflow, evaluation)

### In Progress
- 🔄 Model training and optimization
- 🔄 API endpoint implementation
- 🔄 End-to-end testing

### Planned
- 📋 Documentation and guides
- 📋 Docker deployment
- 📋 Azure cloud integration

## Docker Deployment

```bash
docker-compose up --build
```

## Testing

```bash
pytest tests/ -v
pytest tests/ --cov=src
```

## Documentation

See `docs/architecture.md` for detailed architecture documentation.

## Environment Variables

Create a `.env` file from `.env.example`:
```
OPENAI_API_KEY=<your-key>
AZURE_SUBSCRIPTION_ID=<your-id>
MLFLOW_TRACKING_URI=<tracking-uri>
```

## Contact

For questions or issues, please reach out to:
- **Supervisor**: George Samuel
- **Team**: Group 2

---

**Last Updated**: May 2026
