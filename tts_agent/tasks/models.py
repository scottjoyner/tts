from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from voicebus.schema.events import VoiceBusEvent


TaskStatus = Literal['queued', 'running', 'awaiting_user', 'done', 'failed', 'cancelled']


class SegmentEvent(VoiceBusEvent):
    segment_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskCreate(BaseModel):
    source_segment_id: str
    speaker_user: str
    text: str
    conversation_id: str | None = None
    turn_id: str | None = None
    priority: Literal['critical', 'interactive', 'background'] = 'interactive'


class TaskRecord(BaseModel):
    task_id: str
    source_segment_id: str
    created_ts: str
    speaker_user: str
    text: str
    status: TaskStatus
    conversation_id: str | None = None
    turn_id: str | None = None
    priority: str = 'interactive'
    task_signature: str | None = None
    attempts: int = 0
    result_summary: str | None = None
