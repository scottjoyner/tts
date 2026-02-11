from __future__ import annotations

import asyncio
import time
from collections.abc import Awaitable, Callable

from tts_agent.tts.priority_queue import AudioPriorityQueue


class InterruptiblePlayback:
    def __init__(self, queue: AudioPriorityQueue, broadcaster: Callable[[dict, bytes], Awaitable[None]]) -> None:
        self.queue = queue
        self.broadcaster = broadcaster
        self._stop_event = asyncio.Event()
        self._active_task: str | None = None
        self._ttfb_ms: list[float] = []

    def interrupt(self) -> None:
        self._stop_event.set()

    @property
    def active_task(self) -> str | None:
        return self._active_task

    def avg_ttfb_ms(self) -> float:
        return sum(self._ttfb_ms) / len(self._ttfb_ms) if self._ttfb_ms else 0.0

    async def run(self, synthesize_stream: Callable[[str], Awaitable]):
        while True:
            item = await self.queue.get()
            self._stop_event.clear()
            self._active_task = item.task_id
            start = time.monotonic()
            first = True
            async for chunk, audio in synthesize_stream(item.text):
                if self._stop_event.is_set():
                    await self.broadcaster({'event_type': 'tts_interrupted', 'task_id': item.task_id}, b'')
                    break
                if first:
                    self._ttfb_ms.append((time.monotonic() - start) * 1000.0)
                    first = False
                await self.broadcaster(
                    {
                        'event_type': 'tts_chunk',
                        'task_id': item.task_id,
                        'priority': item.priority,
                        'text': chunk,
                        'bytes': len(audio),
                    },
                    audio,
                )
            self._active_task = None
