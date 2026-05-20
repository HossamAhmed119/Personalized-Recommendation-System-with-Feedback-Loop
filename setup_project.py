import os
import json
from pathlib import Path

PROJECT_NAME = "recommendation-system"

notebook_template = json.dumps({
    "cells": [],
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3"
        }
    },
    "nbformat": 4,
    "nbformat_minor": 4
})

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
    "docs",
    "docs/figures",
    "notebooks",
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
    "docs/figures/.gitkeep": "",

    "notebooks/01_EDA.ipynb": notebook_template,
    "notebooks/02_preprocessing.ipynb": notebook_template,
    "notebooks/03_modeling.ipynb": notebook_template,

    ".gitignore": ".env\ndata/\nlogs/\n__pycache__/\n*.pyc\nvenv/\nsystem_env/",
    "data/raw/.gitkeep": "",
    "data/processed/.gitkeep": "",
    "data/embeddings/.gitkeep": "",
}


def create_project():
    print(f"\n🚀 Setting up project: {PROJECT_NAME}\n")

    # create folders
    for folder in folders:
        path = os.path.join(PROJECT_NAME, folder)
        os.makedirs(path, exist_ok=True)
        print(f"[+] Created folder: {path}")

    print()

    # create files — skip if already exists
    for file_path, content in files.items():
        path = os.path.join(PROJECT_NAME, file_path)
        os.makedirs(os.path.dirname(path), exist_ok=True)

        if Path(path).exists():
            print(f"[SKIP] Already exists: {path}")
            continue

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        print(f"[+] Created file: {path}")

    print("\n🎉 Project structure created successfully!")


if __name__ == "__main__":
    create_project()
