import pandas as pd
from pathlib import Path

from src.utils.config_loader import load_app_config, load_data_config
from src.database.db_manager import DatabaseManager

def setup_initial_database():
    print("Loading configurations...")
    
    project_root = Path(__file__).resolve().parents[2]
    app_config_path = project_root / "configs" / "app_config.yaml"
    data_config_path = project_root / "configs" / "data_config.yaml"
    
    app_config = load_app_config(str(app_config_path))
    data_config = load_data_config(str(data_config_path))
    
    db_path = project_root / app_config['database']['interactions_db']
    
    # Construct path to the processed data to extract users
    # Assuming the train_file contains the users you trained on
    processed_dir = project_root / data_config['paths']['processed_data']
    train_file = data_config['paths']['train_file']
    processed_data_path = processed_dir / train_file
    
    print(f"Initializing database at: {db_path}")
    db = DatabaseManager(str(db_path))
    
    if processed_data_path.exists():
        print(f"Loading users from: {processed_data_path}")
        df = pd.read_parquet(processed_data_path, columns=['user_id'])
        unique_users = df['user_id'].dropna().unique().tolist()
        
        print(f"Seeding {len(unique_users)} old users into the database...")
        db.seed_users(unique_users)
        print("Database seeding completed successfully.")
    else:
        print(f"Warning: Processed data file not found at {processed_data_path}")
        print("Database created, but no users were seeded.")

if __name__ == "__main__":
    setup_initial_database()