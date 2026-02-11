from __future__ import annotations

import asyncio
from dataclasses import dataclass, field

from tts_agent.queue.priorities import rank
from tts_agent.tasks.models import TaskRecord


@dataclass(order=True)
class QueueItem:
    order: tuple[int, float] = field(init=False)
    task: TaskRecord = field(compare=False)

    def __post_init__(self) -> None:
        self.order = (rank(self.task.priority), float(self.task.created_ts.replace('-', '').replace(':', '').replace('T', '').replace('Z', '').replace('.', '')[:14] or 0))


class TaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[QueueItem] = asyncio.PriorityQueue()

    async def put(self, task: TaskRecord) -> None:
        await self._queue.put(QueueItem(task=task))

    async def get(self) -> TaskRecord:
        item = await self._queue.get()
        return item.task

    def qsize(self) -> int:
        return self._queue.qsize()
