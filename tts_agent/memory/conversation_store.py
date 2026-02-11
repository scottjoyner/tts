from __future__ import annotations

import sqlite3
from pathlib import Path


class ConversationStore:
    def __init__(self, db_path: Path, max_turns: int = 10) -> None:
        self.max_turns = max_turns
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS transcript_log(id INTEGER PRIMARY KEY AUTOINCREMENT, conversation_id TEXT, role TEXT, text TEXT, ts TEXT DEFAULT CURRENT_TIMESTAMP)'
        )
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS conversation_summary(conversation_id TEXT PRIMARY KEY, summary TEXT, updated_ts TEXT DEFAULT CURRENT_TIMESTAMP)'
        )
        self._conn.commit()

    def append_turn(self, conversation_id: str, role: str, text: str) -> None:
        self._conn.execute(
            'INSERT INTO transcript_log(conversation_id, role, text) VALUES (?, ?, ?)',
            (conversation_id, role, text),
        )
        self._conn.commit()

    def recent_turns(self, conversation_id: str) -> list[dict]:
        rows = self._conn.execute(
            'SELECT role, text, ts FROM transcript_log WHERE conversation_id = ? ORDER BY id DESC LIMIT ?',
            (conversation_id, self.max_turns),
        ).fetchall()
        return [dict(row) for row in reversed(rows)]

    def get_summary(self, conversation_id: str) -> str:
        row = self._conn.execute(
            'SELECT summary FROM conversation_summary WHERE conversation_id = ?', (conversation_id,)
        ).fetchone()
        return row['summary'] if row else ''

    def upsert_summary(self, conversation_id: str, summary: str) -> None:
        self._conn.execute(
            'INSERT INTO conversation_summary(conversation_id, summary) VALUES (?, ?) '
            'ON CONFLICT(conversation_id) DO UPDATE SET summary = excluded.summary, updated_ts = CURRENT_TIMESTAMP',
            (conversation_id, summary),
        )
        self._conn.commit()
