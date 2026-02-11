from __future__ import annotations

from dataclasses import dataclass

from tts_agent.tasks.models import TaskStatus

_ALLOWED: dict[TaskStatus, set[TaskStatus]] = {
    'queued': {'running', 'done', 'failed', 'cancelled'},
    'running': {'awaiting_user', 'done', 'failed', 'cancelled'},
    'awaiting_user': {'running', 'cancelled', 'done', 'failed'},
    'done': set(),
    'failed': set(),
    'cancelled': set(),
}


@dataclass
class TransitionResult:
    allowed: bool
    reason: str


def can_transition(current: TaskStatus, target: TaskStatus) -> TransitionResult:
    if target in _ALLOWED[current]:
        return TransitionResult(True, f'{current} -> {target}')
    return TransitionResult(False, f'invalid transition: {current} -> {target}')
