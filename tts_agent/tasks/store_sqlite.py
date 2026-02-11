from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from tts_agent.tasks.models import TaskCreate, TaskRecord, TaskStatus
from tts_agent.tasks.state_machine import can_transition
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
                conversation_id TEXT,
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
                stage TEXT DEFAULT 'execution',
                latency_ms REAL DEFAULT 0,
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
                priority TEXT DEFAULT 'task_response',
                audio_path TEXT
            )
            '''
        )
        self._maybe_add_column('tasks', 'conversation_id', 'TEXT')
        self._maybe_add_column('llm_calls', 'stage', "TEXT DEFAULT 'execution'")
        self._maybe_add_column('llm_calls', 'latency_ms', 'REAL DEFAULT 0')
        self._maybe_add_column('tts_outputs', 'priority', "TEXT DEFAULT 'task_response'")
        self._conn.commit()

    def _maybe_add_column(self, table: str, column: str, coldef: str) -> None:
        columns = {row['name'] for row in self._conn.execute(f'PRAGMA table_info({table})').fetchall()}
        if column not in columns:
            self._conn.execute(f'ALTER TABLE {table} ADD COLUMN {column} {coldef}')

    def create_task(self, data: TaskCreate) -> TaskRecord:
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            source_segment_id=data.source_segment_id,
            created_ts=utc_now_iso(),
            speaker_user=data.speaker_user,
            text=data.text,
            status='queued',
            conversation_id=data.conversation_id,
        )
        self._conn.execute(
            'INSERT INTO tasks(task_id, source_segment_id, created_ts, speaker_user, text, status, conversation_id, result_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (
                task.task_id,
                task.source_segment_id,
                task.created_ts,
                task.speaker_user,
                task.text,
                task.status,
                task.conversation_id,
                task.result_summary,
            ),
        )
        self.record_task_event(task.task_id, 'state_changed', '{"to": "queued"}')
        self._conn.commit()
        return task

    def update_task_status(self, task_id: str, status: TaskStatus, result_summary: str | None = None) -> None:
        task = self.get_task(task_id)
        if task is None:
            return
        transition = can_transition(task.status, status)
        if not transition.allowed and task.status != status:
            raise ValueError(transition.reason)
        self._conn.execute(
            'UPDATE tasks SET status = ?, result_summary = COALESCE(?, result_summary) WHERE task_id = ?',
            (status, result_summary, task_id),
        )
        self.record_task_event(task_id, 'state_changed', f'{{"from": "{task.status}", "to": "{status}"}}')
        self._conn.commit()

    def cancel_task(self, task_id: str, reason: str = 'user_request') -> bool:
        task = self.get_task(task_id)
        if not task or task.status in {'done', 'failed', 'cancelled'}:
            return False
        self.update_task_status(task_id, 'cancelled', f'cancelled: {reason}')
        return True

    def get_task(self, task_id: str) -> TaskRecord | None:
        row = self._conn.execute('SELECT * FROM tasks WHERE task_id = ?', (task_id,)).fetchone()
        return TaskRecord(**dict(row)) if row else None

    def list_tasks(self, limit: int = 50) -> list[TaskRecord]:
        rows = self._conn.execute('SELECT * FROM tasks ORDER BY created_ts DESC LIMIT ?', (limit,)).fetchall()
        return [TaskRecord(**dict(row)) for row in rows]

    def get_active_task(self) -> TaskRecord | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('queued', 'running', 'awaiting_user') ORDER BY created_ts ASC LIMIT 1"
        ).fetchone()
        return TaskRecord(**dict(row)) if row else None

    def record_task_event(self, task_id: str, event_type: str, payload_json: str) -> None:
        self._conn.execute(
            'INSERT INTO task_events(task_id, ts, event_type, payload_json) VALUES (?, ?, ?, ?)',
            (task_id, utc_now_iso(), event_type, payload_json),
        )
        self._conn.commit()

    def record_llm_call(
        self,
        task_id: str,
        iteration: int,
        model_name: str,
        prompt: str,
        response: str,
        stage: str = 'execution',
        latency_ms: float = 0,
    ) -> None:
        self._conn.execute(
            'INSERT INTO llm_calls(task_id, iteration, model_name, prompt, response, stage, latency_ms, ts) VALUES (?, ?, ?, ?, ?, ?, ?, ?)',
            (task_id, iteration, model_name, prompt, response, stage, latency_ms, utc_now_iso()),
        )
        self._conn.commit()

    def record_tts_output(self, task_id: str, text: str, audio_path: str | None = None, priority: str = 'task_response') -> None:
        self._conn.execute(
            'INSERT INTO tts_outputs(task_id, ts, text, priority, audio_path) VALUES (?, ?, ?, ?, ?)',
            (task_id, utc_now_iso(), text, priority, audio_path),
        )
        self._conn.commit()

    def task_trace(self, task_id: str) -> dict[str, Any]:
        iterations = [dict(row) for row in self._conn.execute('SELECT * FROM task_events WHERE task_id = ? ORDER BY event_id ASC', (task_id,)).fetchall()]
        llm_calls = [dict(row) for row in self._conn.execute('SELECT * FROM llm_calls WHERE task_id = ? ORDER BY call_id ASC', (task_id,)).fetchall()]
        outputs = [dict(row) for row in self._conn.execute('SELECT * FROM tts_outputs WHERE task_id = ? ORDER BY output_id ASC', (task_id,)).fetchall()]
        return {'task_id': task_id, 'iterations': iterations, 'llm_calls': llm_calls, 'outputs': outputs}

    def stats(self) -> dict[str, Any]:
        cursor = self._conn.cursor()
        counts = {
            status: cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', (status,)).fetchone()[0]
            for status in ('queued', 'running', 'awaiting_user', 'done', 'failed', 'cancelled')
        }
        counts['total'] = cursor.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
        avg_duration = cursor.execute(
            "SELECT AVG((julianday(MAX(ts)) - julianday(MIN(ts))) * 86400.0) FROM task_events GROUP BY task_id"
        ).fetchall()
        counts['avg_task_duration_s'] = float(sum(row[0] for row in avg_duration if row[0] is not None) / len(avg_duration)) if avg_duration else 0.0
        return counts
