"""
Configuration loader for the recommendation system.
Provides generic YAML loading and validated config loaders.
"""

import yaml
from pathlib import Path
from typing import Dict, Any, List, Optional


# ============================================
# GENERIC LOADER
# ============================================
def load_config(config_path: str) -> Dict[str, Any]:
    """
    Load any YAML configuration file.
    
    Args:
        config_path: Path to YAML config file
        
    Returns:
        Configuration dictionary
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config


# ============================================
# DATA CONFIG LOADER
# ============================================
def load_data_config(config_path: str = "configs/data_config.yaml") -> Dict[str, Any]:
    """
    Load and validate data configuration.
    
    Args:
        config_path: Path to data config YAML
        
    Returns:
        Validated data configuration
    """
    config = load_config(config_path)
    
    # Validate required sections
    required_sections = ["dataset", "paths", "preprocessing"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {config_path}")
    
    # Validate dataset config
    if "electronics" not in config.get("dataset", {}):
        raise KeyError("Missing 'dataset.electronics' section")
    
    ds = config["dataset"]["electronics"]
    if "ratings_url" not in ds or "metadata_url" not in ds:
        raise KeyError("Missing ratings_url or metadata_url in dataset config")
    
    return config


# ============================================
# TUNING CONFIG LOADER
# ============================================
def load_tuning_config(config_path: str) -> Dict[str, Any]:
    """
    Load and validate experiment tuning configurations.
    
    Args:
        config_path: Path to tuning config YAML
        
    Returns:
        Validated tuning configuration
    """
    config = load_config(config_path)
    
    required_sections = ["experiment", "search_space"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Invalid tuning config: Missing '{section}' section in {config_path}")
    
    if "name" not in config.get("experiment", {}):
        raise ValueError("Experiment 'name' must be defined in the config.")
    
    return config


# ============================================
# MODEL CONFIG LOADER
# ============================================
def load_model_config(config_path: str = "configs/model_config.yaml") -> Dict[str, Any]:
    """
    Load and validate model configuration.
    
    Args:
        config_path: Path to model config YAML
        
    Returns:
        Validated model configuration
    """
    config = load_config(config_path)
    
    # Validate required sections
    required_sections = ["experiment", "evaluation", "models", "paths"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {config_path}")
    
    # Validate experiment config
    exp = config.get("experiment", {})
    required_exp = ["name", "train_path", "val_path"]
    for key in required_exp:
        if key not in exp:
            raise KeyError(f"Missing 'experiment.{key}' in {config_path}")
    
    # Validate evaluation config
    eval_cfg = config.get("evaluation", {})
    if "k" not in eval_cfg:
        raise KeyError(f"Missing 'evaluation.k' in {config_path}")
    
    # Validate models config
    models_cfg = config.get("models", {})
    if "common" not in models_cfg:
        raise KeyError(f"Missing 'models.common' in {config_path}")
    
    # Validate each model has params
    expected_models = ["ALS", "BPR", "SVD", "ItemKNN"]
    for model_name in expected_models:
        if model_name not in models_cfg:
            raise KeyError(f"Missing 'models.{model_name}' in {config_path}")
    
    return config

# ============================================
# APP CONFIG LOADER
# ============================================
def load_app_config(config_path: str = "configs/app_config.yaml") -> Dict[str, Any]:
    """
    Load and validate the application configuration for the UI and API.
    
    Args:
        config_path: Path to app config YAML
        
    Returns:
        Validated app configuration dictionary
    """
    config = load_config(config_path)
    
    # Validate required sections
    required_sections = ["app", "database", "paths", "ui"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Missing '{section}' section in {config_path}")
            
    # Validate database paths
    db_cfg = config.get("database", {})
    if "interactions_db" not in db_cfg:
        raise KeyError(f"Missing 'database.interactions_db' in {config_path}")
        
    return config

# ============================================
# MAIN (
# ============================================

# ============================================
# MAIN
# ============================================
if __name__ == "__main__":
    try:
        data_config = load_data_config("configs/data_config.yaml")
        print(" Data config loaded successfully.")
        
        tuning_config = load_tuning_config("configs/experiments/exp_001.yaml")
        print(f" Tuning config loaded for experiment: {tuning_config['experiment']['name']}")
        
        model_config = load_model_config("configs/model_config.yaml")
        print(f" Model config loaded for experiment: {model_config['experiment']['name']}")

        app_config = load_app_config("configs/app_config.yaml")
        print(f" App config loaded for: {app_config['app']['name']}")
        
    except Exception as e:
        print(f" Error: {e}")
