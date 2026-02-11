from __future__ import annotations

from voicebus.schema.tasks import Task


class DeadLetterQueue:
    def __init__(self) -> None:
        self.items: list[Task] = []

    def push(self, task: Task) -> None:
        task.status = 'dead_letter'
        self.items.append(task)

    def list(self) -> list[Task]:
        return list(self.items)
