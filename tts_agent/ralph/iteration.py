from __future__ import annotations

from pydantic import BaseModel, Field


class IterationResult(BaseModel):
    iteration: int
    plan: list[str] = Field(default_factory=list)
    next_action: str
    tool_calls: list[dict] = Field(default_factory=list)
    proposed_response_text: str
    completion_check: dict
    confidence: float
