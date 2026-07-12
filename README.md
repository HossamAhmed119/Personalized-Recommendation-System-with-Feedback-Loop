# Amazon Electronics Recommendation System

A production-oriented hybrid recommendation system built on the Amazon Reviews 2023 Electronics dataset. The project covers the complete recommender system lifecycle, from large-scale data ingestion and preprocessing to model training, evaluation, deployment, and an intelligent RAG-powered shopping assistant.

Rather than focusing on a single recommendation algorithm, this project explores the evolution of recommendation systems through multiple modeling generations, rigorous experimentation, and production-ready deployment.

**DEPI Graduation Project · 2026**

**Team:** Ahmed Mohamed Yousef Elkhafef · Mohamed Abdelaziz Kamal · Ahmed Mohamed Abdelaziz · Hossam Ahmed Farouq

**Supervisor:** Eng. George Samuel

---

# Project Overview

This project implements an end-to-end recommendation platform designed to simulate a real production recommendation system.

The pipeline begins with large-scale ingestion of Amazon Electronics interactions using a memory-efficient chunked processing strategy, followed by an extensive preprocessing pipeline that removes noisy interactions, applies iterative K-Core filtering, performs feature engineering, and prepares optimized datasets for multiple recommendation models.

Several recommendation approaches were implemented and compared throughout the project:

* Classical Collaborative Filtering (ALS, BPR, SVD, ItemKNN)
* Content-Based Filtering using TF-IDF
* Neural Collaborative Filtering (NCF)
* A Hybrid Recommendation Model combining NCF and Content-Based predictions

To transform the recommendation engine into a practical shopping assistant, a Retrieval-Augmented Generation (RAG) pipeline was integrated using ChromaDB, Sentence Transformers, and an LLM through OpenRouter, allowing users to receive conversational product recommendations instead of static recommendation lists.

The entire experimentation workflow is tracked using MLflow, hyperparameters are optimized with Optuna, and the final system is deployed through FastAPI and Streamlit.

Every metric reported in this repository is generated directly from the implemented training and evaluation pipelines.

---

# Key Features

## 1. Large-Scale Data Pipeline

* Chunk-based ingestion supporting datasets up to 20 million interactions
* Memory-safe preprocessing pipeline
* Missing value handling
* Duplicate removal
* Spam filtering
* Feature engineering
* Iterative K-Core filtering
* Time-based filtering (2017 onward)
* Sparse matrix generation for recommendation models

---

## 2. Multiple Recommendation Models

The project investigates several recommendation paradigms rather than relying on a single model.

### Classical Collaborative Filtering

Implemented baseline algorithms include:

* ALS
* BPR
* SVD
* ItemKNN

These models establish collaborative filtering baselines using user-item interactions.

### Content-Based Filtering

A TF-IDF based recommendation engine was developed to recommend products according to product descriptions and textual metadata, helping reduce cold-start issues.

### Neural Collaborative Filtering

A PyTorch implementation of Neural Collaborative Filtering captures complex non-linear relationships between users and products, significantly outperforming classical collaborative filtering methods.

### Hybrid Recommendation Model

The final production model combines Neural Collaborative Filtering with Content-Based Filtering using weighted score fusion.

The optimal blending parameter (α = 0.70) was selected using a two-stage evaluation strategy and deployed as the final recommendation engine.

---

## 3. Intelligent Shopping Assistant (RAG)

The recommendation engine is enhanced with a Retrieval-Augmented Generation pipeline that includes:

* ChromaDB vector database
* Sentence Transformers embeddings
* Semantic retrieval
* OpenRouter LLM integration
* Conversational recommendation interface

Instead of only returning products, the assistant explains recommendations and answers shopping-related questions using retrieved product information.

---

## 4. Production Serving Layer

The complete serving architecture includes:

* FastAPI REST API
* Streamlit web interface
* SQLite interaction database
* Production recommendation engine
* Configuration-driven deployment

---

## 5. Experiment Tracking

All experiments are tracked using MLflow.

The project records:

* Training runs
* Model versions
* Hyperparameters
* Evaluation metrics
* Model comparisons

Hyperparameter optimization is performed using Optuna.

---

# Model Performance

| Model                          | HitRate@10 |    NDCG@10 | Notes                      |
| ------------------------------ | ---------: | ---------: | -------------------------- |
| Content-Based (TF-IDF)         |      1.76% |     0.0064 | Standalone model           |
| Neural Collaborative Filtering |      4.55% |     0.0152 | Best standalone model      |
| **Hybrid (α = 0.70)**          |  **4.71%** | **0.0158** | **Final production model** |

The Hybrid model achieved the strongest overall performance by combining behavioral patterns learned by Neural Collaborative Filtering with semantic similarities extracted through Content-Based Filtering.

---

# Dataset

**Source**

Amazon Reviews 2023

Electronics Category

Hou et al.

"Amazon Reviews 2023: A Comprehensive Benchmark Dataset"

---

### Dataset Characteristics

* Supports processing up to **20 million** raw interactions
* Processed using **1 million row chunks**
* Extremely sparse dataset (>99.99%)
* Final modeling matrix:

  * **997,856 Users**
  * **9,880 Products**

After preprocessing, the data undergoes:

* Spam removal
* Iterative K-Core filtering
* Top-N item filtering
* Time-based filtering

---

# System Architecture

```
Raw Amazon Dataset
        │
        ▼
Chunked Data Ingestion
        │
        ▼
Preprocessing & Feature Engineering
        │
        ▼
Recommendation Models
 ├── ALS
 ├── BPR
 ├── SVD
 ├── ItemKNN
 ├── Content-Based
 ├── Neural CF
 └── Hybrid Model
        │
        ▼
MLflow + Optuna
        │
        ▼
Recommendation Engine
        │
        ├── FastAPI
        ├── Streamlit
        ├── SQLite
        └── RAG Assistant
```

---

# Technology Stack

## Data Processing

* pandas
* NumPy
* PyArrow
* SciPy
* scikit-learn

## Recommendation Models

* implicit
* PyTorch

## RAG & NLP

* ChromaDB
* Sentence Transformers
* OpenRouter API

## Experimentation

* MLflow
* Optuna

## Backend

* FastAPI

## Frontend

* Streamlit

## Database

* SQLite

---

# Project Structure

```
recommendation-system/
├── configs/
│   ├── app_config.yaml
│   ├── data_config.yaml
│   └── model_config.yaml
│
├── data/
│   ├── raw/
│   ├── processed/
│   └── embeddings/
│       └── chroma_db/
│
├── models/                      # serialized trained model artifacts
│   ├── best_als_model.pkl
│   ├── best_cb_model.pkl
│   └── best_ncf_model.pt
│
├── mlops/
│   ├── compare_baselines.py
│   ├── evaluate.py
│   ├── evaluate_hybrid.py
│   ├── mlflow_tracking.py
│   ├── train.py
│   ├── train_final.py
│   ├── train_final_cb.py
│   ├── train_final_ncf.py
│   └── tune.py
│
├── notebooks/
│   ├── 01_EDA.ipynb
│   ├── 02_preprocessing.ipynb
│   ├── 03_modeling.ipynb
│   ├── 04_content_based.ipynb
│   └── 05_neural_cf.ipynb
│
├── reports/                     # generated comparison plots/metrics
│
├── src/
│   ├── api/
│   │   └── main.py              # FastAPI service
│   │
│   ├── data_pipeline/
│   │   ├── ingest.py
│   │   ├── preprocess.py
│   │   └── extract_images.py
│   │
│   ├── database/
│   │   └── db_manager.py        # SQLite interaction store
│   │
│   ├── models/
│   │   ├── cf_model.py          # BaseRecommender, ALS, BPR, SVD, ItemKNN
│   │   ├── cb_model.py          # Content-Based (TF-IDF)
│   │   ├── ncf_model.py         # Neural Collaborative Filtering
│   │   ├── hybrid_model.py      # Hybrid fusion
│   │   ├── recommendation_engine.py   # production inference engine
│   │   ├── run_preprocessing.py       # full production preprocessing pipeline
│   │   └── compare.py
│   │
│   ├── rag/
│   │   ├── vector_store.py      # ChromaDB builder
│   │   └── retriever.py         # RAG chat pipeline
│   │
│   └── utils/
│       ├── config_loader.py
│       ├── eda_summary.py
│       └── visualization.py
│
├── app.py                       # Streamlit application (entry point)
├── requirements.txt             # TODO: confirm/generate
└── README.md
```

---

# Getting Started

```bash
# Clone repository
git clone <repository-url>

# Install dependencies
pip install -r requirements.txt

# Configure environment
cp .env.example .env

# Data pipeline
python src/data_pipeline/ingest.py
python src/models/run_preprocessing.py

# Train models
python mlops/train_final.py
python mlops/train_final_cb.py
python mlops/train_final_ncf.py

# Evaluate Hybrid
python mlops/evaluate_hybrid.py

# Launch FastAPI
uvicorn src.api.main:app --reload

# Launch Streamlit
streamlit run app.py

# Open MLflow
mlflow ui
```

---

# Evaluation Methodology

To ensure realistic model evaluation, a time-based train/validation/test split was used instead of random splitting, preventing temporal data leakage.

Hybrid model tuning followed a two-stage strategy:

1. Grid search over candidate α values using a validation sample.
2. Full validation using all users to confirm the optimal blending weight.

Tracked metrics include:

* HitRate@10
* NDCG@10
* Precision
* Adjusted Precision
* Recall
* Mean Reciprocal Rank (MRR)

---

# Future Work

The current system provides a strong production baseline, while several improvements remain for future research.

### Engineering Improvements

* Automated pipeline testing
* Diversity-aware recommendation
* Popularity re-ranking
* Continuous deployment pipeline

### Research Improvements

* Sequential recommendation models (SASRec, GRU4Rec)
* Transformer-based recommendation
* Ranking-aware loss functions
* Online A/B testing
* Reinforcement learning for recommendation

---

# References

* Gomez-Uribe, C. A., & Hunt, N. (2015). *The Netflix Recommender System.*
* Hou, Y. et al. (2024). *Amazon Reviews 2023: A Comprehensive Benchmark Dataset.*
* Huang, D. et al. (2023). *Revisiting Neural Collaborative Filtering.*



**Graduation Project — Digital Egypt Pioneers Initiative (DEPI)**

**Supervisor:** Eng. George Samuel
