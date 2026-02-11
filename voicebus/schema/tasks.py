from __future__ import annotations

import time
import uuid
from typing import Literal

from pydantic import BaseModel, Field

TaskPriority = Literal['critical', 'interactive', 'background']
TaskStatus = Literal['queued', 'leased', 'running', 'completed', 'failed', 'cancelled', 'dead_letter']


class Task(BaseModel):
    schema_version: Literal['3.0'] = '3.0'
    task_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    turn_id: str
    task_signature: str
    text: str
    speaker_user: str
    priority: TaskPriority = 'interactive'
    status: TaskStatus = 'queued'
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))
    max_attempts: int = 3


class TaskRun(BaseModel):
    run_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    task_id: str
    lease_owner: str
    lease_expires_at_ms: int
    heartbeat_at_ms: int
    status: TaskStatus = 'leased'
    error: str | None = None


class AgentStep(BaseModel):
    step_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    run_id: str
    task_id: str
    agent_name: str
    status: Literal['started', 'result']
    content: str
    artifact_uri: str | None = None
