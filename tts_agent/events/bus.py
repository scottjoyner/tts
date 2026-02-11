from __future__ import annotations

import asyncio
from collections import defaultdict
from collections.abc import AsyncIterator
from typing import Any


class EventBus:
    def __init__(self) -> None:
        self._subscribers: dict[str, list[asyncio.Queue[dict[str, Any]]]] = defaultdict(list)

    async def publish(self, topic: str, payload: dict[str, Any]) -> None:
        for queue in self._subscribers.get(topic, []):
            await queue.put(payload)

    async def subscribe(self, topic: str) -> AsyncIterator[dict[str, Any]]:
        queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._subscribers[topic].append(queue)
        try:
            while True:
                yield await queue.get()
        finally:
            self._subscribers[topic].remove(queue)
