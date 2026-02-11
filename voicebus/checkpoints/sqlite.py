from __future__ import annotations

import sqlite3
from pathlib import Path


class SQLiteCheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS checkpoints (consumer TEXT PRIMARY KEY, cursor INTEGER NOT NULL DEFAULT 0, updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP)'
        )
        self._conn.commit()

    def get(self, consumer: str) -> int:
        row = self._conn.execute('SELECT cursor FROM checkpoints WHERE consumer = ?', (consumer,)).fetchone()
        return int(row[0]) if row else 0

    def set(self, consumer: str, cursor: int) -> None:
        self._conn.execute(
            'INSERT INTO checkpoints(consumer, cursor) VALUES (?, ?) ON CONFLICT(consumer) DO UPDATE SET cursor = excluded.cursor, updated_at = CURRENT_TIMESTAMP',
            (consumer, cursor),
        )
        self._conn.commit()
