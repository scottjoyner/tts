from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


TaskStatus = Literal['queued', 'running', 'awaiting_user', 'done', 'failed', 'cancelled']


class SegmentEvent(BaseModel):
    event_type: str
    event_id: str | None = None
    segment_id: str | None = None
    transcript_final: str | None = None
    transcript_partial: str | None = None
    triggered: bool = False
    is_command: bool = False
    authenticated: bool = False
    authenticated_user: str | None = None
    speaker_candidate: str | None = None
    speaker_score: float | None = None
    actionability: dict[str, Any] = Field(default_factory=dict)
    conversation_id: str | None = None
    start_ts: float | None = None
    end_ts: float | None = None
    raw: dict[str, Any] = Field(default_factory=dict)


class TaskCreate(BaseModel):
    source_segment_id: str
    speaker_user: str
    text: str
    conversation_id: str | None = None


class TaskRecord(BaseModel):
    task_id: str
    source_segment_id: str
    created_ts: str
    speaker_user: str
    text: str
    status: TaskStatus
    conversation_id: str | None = None
    result_summary: str | None = None
