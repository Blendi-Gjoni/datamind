import json
import sqlite3
from pathlib import Path

import pandas as pd

class DataStore:
    def __init__(self, db_path="store.db", data_dir="data/"):
        self.db_path = db_path
        self.data_dir = Path(data_dir)
        self.data_dir.mkdir(exist_ok=True)
        self.init_db()

    def init_db(self):
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS datasets (
                    dataset_id  TEXT PRIMARY KEY,
                    file_path   TEXT NOT NULL,
                    schema      TEXT NOT NULL,
                    metadata    TEXT NOT NULL,
                    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
                )
            """)
    def save(self, dataset_id: str, df: pd.DataFrame, schema: dict, metadata: dict):
        file_path = self.data_dir / f"{dataset_id}.parquet"
        df.to_parquet(file_path, index=False)
        with sqlite3.connect(self.db_path) as conn:
            conn.execute(
                "INSERT INTO datasets (dataset_id, file_path, schema, metadata) VALUES (?, ?, ?, ?)",
                (dataset_id, str(file_path), json.dumps(schema), json.dumps(metadata))
            )

    def get_schema(self, dataset_id: str) -> dict:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT schema, metadata FROM datasets WHERE dataset_id = ?",
                (dataset_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Dataset {dataset_id} not found")
        return json.loads(row[0]), json.loads(row[1])

    def get_dataframe(self, dataset_id: str, columns: list = None) -> pd.DataFrame:
        with sqlite3.connect(self.db_path) as conn:
            row = conn.execute(
                "SELECT file_path FROM datasets WHERE dataset_id = ?",
                (dataset_id,)
            ).fetchone()
        if not row:
            raise KeyError(f"Dataset {dataset_id} not found")
        return pd.read_parquet(row[0], columns=columns)
