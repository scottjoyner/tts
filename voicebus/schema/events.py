from __future__ import annotations

import time
import uuid
from typing import Any, Literal

from pydantic import BaseModel, Field

from voicebus.tracing.trace import TraceContext

SchemaVersion = Literal['3.0']

EventType = Literal[
    'conversation_start',
    'conversation_end',
    'turn_start',
    'turn_update',
    'turn_final',
    'segment_final',
    'pipeline_health',
    'task_created',
    'task_started',
    'task_updated',
    'task_completed',
    'task_failed',
    'task_cancelled',
    'agent_step_started',
    'agent_step_result',
    'model_routing_decision',
    'tts_started',
    'tts_chunk',
    'tts_finished',
    'tts_interrupted',
    'checkpoint',
    'error',
]


class SpeakerContext(BaseModel):
    authenticated: bool = False
    authenticated_user: str | None = None
    candidate: str | None = None
    score: float | None = None


class TriggerContext(BaseModel):
    triggered: bool = False
    phrase: str | None = None


class Actionability(BaseModel):
    actionable: bool = False
    confidence: float = 0.0
    reason: str = 'not_evaluated'


class VoiceBusEvent(BaseModel):
    schema_version: SchemaVersion = '3.0'
    event_type: EventType
    event_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    timestamp_ms: int = Field(default_factory=lambda: int(time.time() * 1000))

    conversation_id: str | None = None
    turn_id: str | None = None
    task_id: str | None = None
    run_id: str | None = None
    step_id: str | None = None

    text: str | None = None
    transcript_partial: str | None = None
    transcript_final: str | None = None

    speaker: SpeakerContext = Field(default_factory=SpeakerContext)
    trigger_context: TriggerContext = Field(default_factory=TriggerContext)
    actionability: Actionability = Field(default_factory=Actionability)

    trace: TraceContext = Field(default_factory=TraceContext)
    payload: dict[str, Any] = Field(default_factory=dict)

    def ensure_actionability(self, classifier_actionable: bool = False) -> None:
        gate = self.speaker.authenticated and (self.trigger_context.triggered or classifier_actionable)
        if gate:
            self.actionability.actionable = True
            if self.actionability.confidence == 0:
                self.actionability.confidence = 0.95 if self.trigger_context.triggered else 0.75
            if self.actionability.reason == 'not_evaluated':
                self.actionability.reason = 'authenticated+trigger_or_classifier'
        else:
            self.actionability.actionable = False
            if self.actionability.reason == 'not_evaluated':
                self.actionability.reason = 'auth_or_trigger_failed'
