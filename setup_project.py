import os

PROJECT_NAME = "recommendation-system"

folders = [
    "configs/experiments",
    "data/raw",
    "data/processed",
    "data/embeddings",
    "src/utils",
    "src/data_pipeline",
    "src/models",
    "src/rag",
    "src/services",
    "src/api",
    "mlops",
    "tests",
    "docs"
]

files = {
    "README.md": "# Recommendation System ",
    "requirements.txt": "",
    ".env.example": "",
    "Dockerfile": "",
    "docker-compose.yml": "",

    "configs/app_config.yaml": "",
    "configs/data_config.yaml": "",
    "configs/model_config.yaml": "",
    "configs/training_config.yaml": "",

    "configs/experiments/exp_001.yaml": "",
    "configs/experiments/exp_002.yaml": "",

    "src/utils/config_loader.py": "",
    "src/utils/logger.py": "",
    "src/utils/helpers.py": "",

    "src/data_pipeline/ingest.py": "",
    "src/data_pipeline/preprocess.py": "",
    "src/data_pipeline/features.py": "",

    "src/models/cf_model.py": "",
    "src/models/deep_model.py": "",
    "src/models/llm_reranker.py": "",

    "src/rag/embedder.py": "",
    "src/rag/vector_store.py": "",
    "src/rag/retriever.py": "",

    "src/services/recommender_service.py": "",

    "src/api/main.py": "",

    "src/__init__.py": "# Auto-generated package initializer",
    "src/utils/__init__.py": "# Auto-generated package initializer",
    "src/data_pipeline/__init__.py": "# Auto-generated package initializer",
    "src/models/__init__.py": "# Auto-generated package initializer",
    "src/rag/__init__.py": "# Auto-generated package initializer",
    "src/services/__init__.py": "# Auto-generated package initializer",
    "src/api/__init__.py": "# Auto-generated package initializer",

    "mlops/train.py": "",
    "mlops/evaluate.py": "",
    "mlops/mlflow_tracking.py": "",

    "docs/architecture.md": "",

    ".gitignore": ".env\ndata/\nlogs/\n__pycache__/\n*.pyc\nvenv/",
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "data/embeddings/.gitkeep": ""


}

def create_project():
    # create folders
    for folder in folders:
        path = os.path.join(PROJECT_NAME, folder)
        os.makedirs(path, exist_ok=True)
        print(f"[+] Created folder: {path}")

    # create files
    for file_path, content in files.items():
        path = os.path.join(PROJECT_NAME, file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Created file: {path}")

    print("\n🎉 Project structure created successfully!")

if __name__ == "__main__":
    create_project()