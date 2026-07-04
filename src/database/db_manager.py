import sqlite3
from pathlib import Path
from typing import List, Tuple

class DatabaseManager:
    def __init__(self, db_path: str):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize_tables()

    def _get_connection(self) -> sqlite3.Connection:
        return sqlite3.connect(str(self.db_path))

    def _initialize_tables(self):
        with self._get_connection() as conn:
            cursor = conn.cursor()
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS users (
                    user_id TEXT PRIMARY KEY,
                    is_new BOOLEAN DEFAULT 0,
                    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            ''')
            
            cursor.execute('''
                CREATE TABLE IF NOT EXISTS interactions (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    user_id TEXT,
                    product_id TEXT,
                    interaction_type TEXT,
                    timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (user_id) REFERENCES users (user_id)
                )
            ''')
            conn.commit()

    def seed_users(self, users_list: List[str]):
        """Seed a list of existing users into the database."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.executemany(
                'INSERT OR IGNORE INTO users (user_id, is_new) VALUES (?, ?)',
                [(str(user), False) for user in users_list]
            )
            conn.commit()

    def add_new_user(self, user_id: str):
        """Register a new user (Cold Start)."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                'INSERT OR IGNORE INTO users (user_id, is_new) VALUES (?, ?)',
                (user_id, True)
            )
            conn.commit()

    def record_interaction(self, user_id: str, product_id: str, interaction_type: str):
        """Record a user's interaction with a product."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(
                '''
                INSERT INTO interactions (user_id, product_id, interaction_type) 
                VALUES (?, ?, ?)
                ''',
                (user_id, product_id, interaction_type)
            )
            conn.commit()

    def get_all_users(self) -> List[Tuple]:
        """Fetch all users for the frontend dropdown."""
        with self._get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute('SELECT user_id, is_new FROM users ORDER BY created_at DESC')
            return cursor.fetchall()