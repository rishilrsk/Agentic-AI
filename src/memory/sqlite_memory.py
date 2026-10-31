"""
SQLite Persistent Application Memory.
Maintains audit logs of analysis queries, generated SQL, execution metrics, and generated insights.
"""

import sqlite3
from datetime import datetime
from typing import Any
from pathlib import Path

from config import settings
from src.utils.logger import logger


class SQLiteMemory:
    """Manages persistent analysis history in a local SQLite database."""

    def __init__(self, db_path: Path = None):
        self.db_path = db_path or settings.get_sqlite_path()
        self._init_db()

    def _get_connection(self) -> sqlite3.Connection:
        conn = sqlite3.connect(str(self.db_path))
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self._get_connection() as conn:
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS analysis_history (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    dataset_name TEXT,
                    user_query TEXT NOT NULL,
                    sql_query TEXT,
                    execution_success BOOLEAN,
                    row_count INTEGER,
                    execution_time_ms REAL,
                    insights TEXT
                )
                """
            )
            conn.commit()
        logger.info(f"Initialized SQLite memory database at {self.db_path}")

    def save_analysis(
        self,
        dataset_name: str,
        user_query: str,
        sql_query: str,
        execution_success: bool,
        row_count: int,
        execution_time_ms: float,
        insights: str,
    ) -> int:
        """Insert a completed analysis session into SQLite history."""
        ts = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                INSERT INTO analysis_history (
                    timestamp, dataset_name, user_query, sql_query,
                    execution_success, row_count, execution_time_ms, insights
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts,
                    dataset_name,
                    user_query,
                    sql_query,
                    execution_success,
                    row_count,
                    execution_time_ms,
                    insights,
                ),
            )
            conn.commit()
            return cursor.lastrowid

    def get_recent_history(self, limit: int = 20) -> list[dict[str, Any]]:
        """Retrieve recent analysis runs ordered by most recent first."""
        with self._get_connection() as conn:
            cursor = conn.execute(
                """
                SELECT * FROM analysis_history
                ORDER BY id DESC
                LIMIT ?
                """,
                (limit,),
            )
            rows = cursor.fetchall()
            return [dict(row) for row in rows]

    def clear_history(self):
        """Clear all stored analysis history."""
        with self._get_connection() as conn:
            conn.execute("DELETE FROM analysis_history")
            conn.commit()
