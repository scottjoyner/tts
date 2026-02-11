from __future__ import annotations

import time
import uuid
from pydantic import BaseModel, Field


class TraceContext(BaseModel):
    trace_id: str = Field(default_factory=lambda: uuid.uuid4().hex)
    span_id: str = Field(default_factory=lambda: uuid.uuid4().hex[:16])
    parent_span_id: str | None = None
    created_at_ms: int = Field(default_factory=lambda: int(time.time() * 1000))


def child_trace(parent: TraceContext) -> TraceContext:
    return TraceContext(trace_id=parent.trace_id, parent_span_id=parent.span_id)
