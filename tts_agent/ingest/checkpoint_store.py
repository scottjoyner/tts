from __future__ import annotations

import hashlib
import sqlite3
from pathlib import Path


class IngestCheckpointStore:
    def __init__(self, db_path: Path) -> None:
        self._conn = sqlite3.connect(db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS ingest_checkpoints(source TEXT PRIMARY KEY, last_event_id TEXT, jsonl_offset INTEGER DEFAULT 0)'
        )
        self._conn.execute(
            'CREATE TABLE IF NOT EXISTS ingest_dedupe(event_hash TEXT PRIMARY KEY, first_seen_ts TEXT DEFAULT CURRENT_TIMESTAMP)'
        )
        self._conn.commit()

    def last_event_id(self, source: str) -> str | None:
        row = self._conn.execute('SELECT last_event_id FROM ingest_checkpoints WHERE source = ?', (source,)).fetchone()
        return row['last_event_id'] if row and row['last_event_id'] else None

    def jsonl_offset(self, source: str) -> int:
        row = self._conn.execute('SELECT jsonl_offset FROM ingest_checkpoints WHERE source = ?', (source,)).fetchone()
        return int(row['jsonl_offset']) if row else 0

    def update_checkpoint(self, source: str, event_id: str | None = None, offset: int | None = None) -> None:
        current_offset = self.jsonl_offset(source)
        current_id = self.last_event_id(source)
        self._conn.execute(
            'INSERT INTO ingest_checkpoints(source, last_event_id, jsonl_offset) VALUES (?, ?, ?) '
            'ON CONFLICT(source) DO UPDATE SET last_event_id = excluded.last_event_id, jsonl_offset = excluded.jsonl_offset',
            (source, event_id if event_id is not None else current_id, offset if offset is not None else current_offset),
        )
        self._conn.commit()

    def seen_event(self, payload_json: str) -> bool:
        event_hash = hashlib.sha256(payload_json.encode('utf-8')).hexdigest()
        row = self._conn.execute('SELECT event_hash FROM ingest_dedupe WHERE event_hash = ?', (event_hash,)).fetchone()
        if row:
            return True
        self._conn.execute('INSERT INTO ingest_dedupe(event_hash) VALUES (?)', (event_hash,))
        self._conn.commit()
        return False
