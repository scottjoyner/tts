from __future__ import annotations

import sqlite3
import uuid
from pathlib import Path
from typing import Any

from tts_agent.queue.dedupe import make_signature
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
        cur = self._conn.cursor()
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS turns (
                turn_id TEXT PRIMARY KEY,
                conversation_id TEXT NOT NULL,
                transcript TEXT,
                created_ts TEXT NOT NULL
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS tasks (
                task_id TEXT PRIMARY KEY,
                source_segment_id TEXT NOT NULL,
                created_ts TEXT NOT NULL,
                speaker_user TEXT NOT NULL,
                text TEXT NOT NULL,
                status TEXT NOT NULL,
                conversation_id TEXT,
                turn_id TEXT,
                priority TEXT NOT NULL DEFAULT 'interactive',
                task_signature TEXT,
                attempts INTEGER NOT NULL DEFAULT 0,
                lease_owner TEXT,
                lease_expires_at_ms INTEGER,
                result_summary TEXT
            )
            '''
        )
        cur.execute('CREATE UNIQUE INDEX IF NOT EXISTS idx_task_dedupe ON tasks(turn_id, task_signature)')
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS task_runs (
                run_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                lease_owner TEXT NOT NULL,
                status TEXT NOT NULL,
                heartbeat_at_ms INTEGER NOT NULL,
                created_ts TEXT NOT NULL,
                updated_ts TEXT NOT NULL,
                error TEXT
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS agent_steps (
                step_id TEXT PRIMARY KEY,
                run_id TEXT NOT NULL,
                task_id TEXT NOT NULL,
                agent_name TEXT NOT NULL,
                status TEXT NOT NULL,
                content TEXT NOT NULL,
                artifact_uri TEXT,
                ts TEXT NOT NULL
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS artifacts (
                artifact_id TEXT PRIMARY KEY,
                task_id TEXT NOT NULL,
                run_id TEXT,
                kind TEXT NOT NULL,
                path TEXT,
                content TEXT,
                ts TEXT NOT NULL
            )
            '''
        )
        cur.execute(
            '''
            CREATE TABLE IF NOT EXISTS checkpoints (
                checkpoint_id TEXT PRIMARY KEY,
                consumer TEXT NOT NULL,
                cursor TEXT NOT NULL,
                ts TEXT NOT NULL
            )
            '''
        )
        cur.execute(
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
        cur.execute(
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
        cur.execute(
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
        self._conn.commit()

    def create_task(self, data: TaskCreate) -> TaskRecord:
        turn_id = data.turn_id or data.source_segment_id
        signature = make_signature(turn_id, data.text)
        existing = self._conn.execute(
            'SELECT * FROM tasks WHERE turn_id = ? AND task_signature = ? LIMIT 1', (turn_id, signature)
        ).fetchone()
        if existing:
            return TaskRecord(**dict(existing))
        task = TaskRecord(
            task_id=str(uuid.uuid4()),
            source_segment_id=data.source_segment_id,
            created_ts=utc_now_iso(),
            speaker_user=data.speaker_user,
            text=data.text,
            status='queued',
            conversation_id=data.conversation_id,
            turn_id=turn_id,
            priority=data.priority,
            task_signature=signature,
        )
        self._conn.execute(
            'INSERT INTO tasks(task_id, source_segment_id, created_ts, speaker_user, text, status, conversation_id, turn_id, priority, task_signature, attempts, result_summary) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)',
            (
                task.task_id,
                task.source_segment_id,
                task.created_ts,
                task.speaker_user,
                task.text,
                task.status,
                task.conversation_id,
                task.turn_id,
                task.priority,
                task.task_signature,
                task.attempts,
                task.result_summary,
            ),
        )
        self.record_task_event(task.task_id, 'state_changed', '{"to": "queued"}')
        self._conn.commit()
        return task

    def lease_next_task(self, lease_owner: str, lease_expires_at_ms: int) -> TaskRecord | None:
        row = self._conn.execute(
            """
            SELECT * FROM tasks
            WHERE status = 'queued'
            ORDER BY CASE priority WHEN 'critical' THEN 0 WHEN 'interactive' THEN 1 ELSE 2 END, created_ts ASC
            LIMIT 1
            """
        ).fetchone()
        if not row:
            return None
        task = TaskRecord(**dict(row))
        self._conn.execute(
            'UPDATE tasks SET status = ?, lease_owner = ?, lease_expires_at_ms = ?, attempts = attempts + 1 WHERE task_id = ?',
            ('running', lease_owner, lease_expires_at_ms, task.task_id),
        )
        self._conn.execute(
            'INSERT INTO task_runs(run_id, task_id, lease_owner, status, heartbeat_at_ms, created_ts, updated_ts) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (str(uuid.uuid4()), task.task_id, lease_owner, 'running', lease_expires_at_ms, utc_now_iso(), utc_now_iso()),
        )
        self._conn.commit()
        return self.get_task(task.task_id)

    def heartbeat(self, task_id: str, lease_expires_at_ms: int) -> None:
        self._conn.execute(
            'UPDATE tasks SET lease_expires_at_ms = ? WHERE task_id = ?',
            (lease_expires_at_ms, task_id),
        )
        self._conn.execute(
            'UPDATE task_runs SET heartbeat_at_ms = ?, updated_ts = ? WHERE task_id = ? AND status = ?',
            (lease_expires_at_ms, utc_now_iso(), task_id, 'running'),
        )
        self._conn.commit()

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

    def move_to_dlq(self, task_id: str, reason: str) -> None:
        self._conn.execute('UPDATE tasks SET status = ? WHERE task_id = ?', ('failed', task_id))
        self.record_task_event(task_id, 'task_dlq', f'{{"reason": "{reason}"}}')
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

    def list_tasks(self, limit: int = 50, status: str | None = None) -> list[TaskRecord]:
        if status:
            rows = self._conn.execute(
                'SELECT * FROM tasks WHERE status = ? ORDER BY created_ts DESC LIMIT ?', (status, limit)
            ).fetchall()
        else:
            rows = self._conn.execute('SELECT * FROM tasks ORDER BY created_ts DESC LIMIT ?', (limit,)).fetchall()
        return [TaskRecord(**dict(row)) for row in rows]

    def get_active_task(self) -> TaskRecord | None:
        row = self._conn.execute(
            "SELECT * FROM tasks WHERE status IN ('queued', 'running', 'awaiting_user') ORDER BY created_ts ASC LIMIT 1"
        ).fetchone()
        return TaskRecord(**dict(row)) if row else None

    def record_agent_step(self, run_id: str, task_id: str, agent_name: str, status: str, content: str) -> str:
        step_id = str(uuid.uuid4())
        self._conn.execute(
            'INSERT INTO agent_steps(step_id, run_id, task_id, agent_name, status, content, ts) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (step_id, run_id, task_id, agent_name, status, content, utc_now_iso()),
        )
        self._conn.commit()
        return step_id

    def record_artifact(self, task_id: str, kind: str, content: str, run_id: str | None = None, path: str | None = None) -> str:
        artifact_id = str(uuid.uuid4())
        self._conn.execute(
            'INSERT INTO artifacts(artifact_id, task_id, run_id, kind, path, content, ts) VALUES (?, ?, ?, ?, ?, ?, ?)',
            (artifact_id, task_id, run_id, kind, path, content, utc_now_iso()),
        )
        self._conn.commit()
        return artifact_id

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
        events = [
            dict(row)
            for row in self._conn.execute('SELECT * FROM task_events WHERE task_id = ? ORDER BY event_id ASC', (task_id,)).fetchall()
        ]
        llm_calls = [
            dict(row)
            for row in self._conn.execute('SELECT * FROM llm_calls WHERE task_id = ? ORDER BY call_id ASC', (task_id,)).fetchall()
        ]
        outputs = [
            dict(row)
            for row in self._conn.execute('SELECT * FROM tts_outputs WHERE task_id = ? ORDER BY output_id ASC', (task_id,)).fetchall()
        ]
        steps = [
            dict(row)
            for row in self._conn.execute('SELECT * FROM agent_steps WHERE task_id = ? ORDER BY ts ASC', (task_id,)).fetchall()
        ]
        runs = [dict(row) for row in self._conn.execute('SELECT * FROM task_runs WHERE task_id = ? ORDER BY created_ts ASC', (task_id,)).fetchall()]
        return {'task_id': task_id, 'iterations': events, 'llm_calls': llm_calls, 'outputs': outputs, 'runs': runs, 'agent_steps': steps}

    def stats(self) -> dict[str, Any]:
        cursor = self._conn.cursor()
        counts = {
            status: cursor.execute('SELECT COUNT(*) FROM tasks WHERE status = ?', (status,)).fetchone()[0]
            for status in ('queued', 'running', 'awaiting_user', 'done', 'failed', 'cancelled')
        }
        counts['total'] = cursor.execute('SELECT COUNT(*) FROM tasks').fetchone()[0]
        return counts
