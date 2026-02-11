from __future__ import annotations

from dataclasses import dataclass, field

CANCEL_TERMS = {'stop', 'cancel', 'nevermind', 'never mind'}
URGENT_BARGE_IN_TERMS = {'stop', 'wait', 'new request'}


@dataclass
class ActiveTaskRegistry:
    foreground_task_id: str | None = None
    running: set[str] = field(default_factory=set)

    def mark_running(self, task_id: str) -> None:
        self.running.add(task_id)
        if self.foreground_task_id is None:
            self.foreground_task_id = task_id

    def mark_done(self, task_id: str) -> None:
        self.running.discard(task_id)
        if self.foreground_task_id == task_id:
            self.foreground_task_id = next(iter(self.running), None)


def is_cancel_intent(text: str) -> bool:
    candidate = text.strip().lower()
    return any(term in candidate for term in CANCEL_TERMS)


def is_urgent_barge_in(text: str) -> bool:
    candidate = text.strip().lower()
    return any(term in candidate for term in URGENT_BARGE_IN_TERMS)
