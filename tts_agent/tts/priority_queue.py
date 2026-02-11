from __future__ import annotations

import asyncio
from dataclasses import dataclass, field
from itertools import count

_PRIORITIES = {'system_critical': 0, 'task_response': 1, 'backchannel': 2}
_counter = count()


@dataclass(order=True)
class PrioritizedAudio:
    priority_value: int
    order: int
    text: str = field(compare=False)
    task_id: str = field(compare=False)
    priority: str = field(compare=False)


class AudioPriorityQueue:
    def __init__(self) -> None:
        self._queue: asyncio.PriorityQueue[PrioritizedAudio] = asyncio.PriorityQueue()

    async def put(self, task_id: str, text: str, priority: str = 'task_response') -> None:
        await self._queue.put(
            PrioritizedAudio(
                priority_value=_PRIORITIES.get(priority, 10),
                order=next(_counter),
                text=text,
                task_id=task_id,
                priority=priority,
            )
        )

    async def get(self) -> PrioritizedAudio:
        return await self._queue.get()

    def qsize(self) -> int:
        return self._queue.qsize()
