import yaml
from pathlib import Path

def load_config(config_path: str) -> dict:
    """
    Generic function to load any YAML configuration file.
    """
    if not Path(config_path).exists():
        raise FileNotFoundError(f"Config not found: {config_path}")
    
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)
    
    return config

def load_tuning_config(config_path: str) -> dict:
    """
    Specific function to load and validate experiment tuning configurations.
    """
    # Use the generic loader to read the file
    config = load_config(config_path)
    
    # Validate that the necessary tuning sections exist
    required_sections = ["experiment", "search_space"]
    for section in required_sections:
        if section not in config:
            raise KeyError(f"Invalid tuning config: Missing '{section}' section in {config_path}")
            
    # 3. You can add deeper validation here if needed
    if "name" not in config.get("experiment", {}):
        raise ValueError("Experiment 'name' must be defined in the config.")
        
    return config


if __name__ == "__main__":
    try:
        # Testing the data config
        data_config = load_config("configs/data_config.yaml")
        print("Data config loaded successfully.")
        
        # Testing the tuning config
        tuning_config = load_tuning_config("configs/experiments/exp_001.yaml")
        print(f"Tuning config loaded for experiment: {tuning_config['experiment']['name']}")
        
    except Exception as e:
        print(f"Error: {e}")