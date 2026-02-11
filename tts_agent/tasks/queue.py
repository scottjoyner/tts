from __future__ import annotations

import asyncio

from tts_agent.tasks.models import TaskRecord


class TaskQueue:
    def __init__(self) -> None:
        self._queue: asyncio.Queue[TaskRecord] = asyncio.Queue()

    async def put(self, task: TaskRecord) -> None:
        await self._queue.put(task)

    async def get(self) -> TaskRecord:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()
