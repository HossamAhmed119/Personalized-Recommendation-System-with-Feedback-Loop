import yaml
from pathlib import Path


def load_config(config_path: str) -> dict:
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config
    
config = load_config("configs/data_config.yaml")

print(config["dataset"]["electronics"]["ratings_url"])