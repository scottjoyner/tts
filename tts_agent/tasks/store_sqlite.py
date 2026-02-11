from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from tts_agent.tasks.models import TaskCreate, TaskRecord
from tts_agent.utils.time import utc_now_iso


class TaskStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        self._init_db()

    def _init_db(self) -> None:
        cursor = self._conn.cursor()
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                source_segment_id TEXT NOT NULL,
                created_ts TEXT NOT NULL,
                speaker_user TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                result_summary TEXT
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS task_events (
                event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload_json TEXT NOT NULL
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS llm_calls (
                call_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                iteration INTEGER NOT NULL,
                model_name TEXT NOT NULL,
                prompt TEXT NOT NULL,
                response TEXT NOT NULL,
                ts TEXT NOT NULL
            )
            '''
        )
        cursor.execute(
            '''
            CREATE TABLE IF NOT EXISTS tts_outputs (
                output_id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_id TEXT NOT NULL,
                ts TEXT NOT NULL,
                text TEXT NOT NULL,
                audio_path TEXT
            )
            '''
        )
        self._conn.commit()

    def create_task(self, data: TaskCreate) -> TaskRecord:
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            source_segment_id=data.source_segment_id,
            created_ts=utc_now_iso(),
            speaker_user=data.speaker_user,
            text=data.text,
            status='queued',
        )
        self._conn.execute(
            'INSERT INTO tasks(task_id, source_segment_id, created_ts, speaker_user, text, status, result_summary) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (
                task.task_id,
                task.source_segment_id,
                task.created_ts,
                task.speaker_user,
                task.text,
                task.status,
                task.result_summary,
            ),
        )
        self._conn.commit()
        return task

    def update_task_status(self, task_id: str, status: str, result_summary: str | None = None) -> None:
        self._conn.execute(
            'UPDATE tasks SET status = ?, result_summary = ? WHERE task_id = ?',
            (status, result_summary, task_id),
        )
        self._conn.commit()

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self._conn.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,)).fetchone()
        return TaskRecord(**dict(row)) if row else None

    def list_tasks(self, limit: int = 50) -> list[TaskRecord]:
        rows = self._conn.execute('SELECT * FROM tasks ORDER BY created_ts DESC LIMIT ?', (limit,)).fetchall()
        return [TaskRecord(**dict(row)) for row in rows]

    def record_task_event(self, task_id: str, event_type: str, payload_json: str) -> None:
        self._conn.execute(
            'INSERT INTO task_events(task_id, ts, event_type, payload_json) VALUES (?, ?, ?, ?)',
            (task_id, utc_now_iso(), event_type, payload_json),
        )
        self._conn.commit()

    def record_llm_call(self, task_id: str, iteration: int, model_name: str, prompt: str, response: str) -> None:
        self._conn.execute(
            'INSERT INTO llm_calls(task_id, iteration, model_name, prompt, response, ts) VALUES (?, ?, ?, ?, ?, ?)',
            (task_id, iteration, model_name, prompt, response, utc_now_iso()),
        )
        self._conn.commit()

    def record_tts_output(self, task_id: str, text: str, audio_path: str | None = None) -> None:
        self._conn.execute(
            'INSERT INTO tts_outputs(task_id, ts, text, audio_path) VALUES (?, ?, ?, ?)',
            (task_id, utc_now_iso(), text, audio_path),
        )
        self._conn.commit()

    def stats(self) -> dict[str, Any]:
        cursor = self._conn.cursor()
        counts = {
            status: cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', (status,)).fetchone()[0]
            for status in ('queued', 'running', 'done', 'failed')
        }
        counts['total'] = cursor.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
        return counts
