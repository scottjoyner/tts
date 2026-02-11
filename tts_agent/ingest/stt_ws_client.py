from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import websockets

from tts_agent.ingest.normalizer import normalize_event

logger = logging.getLogger(__name__)


class STTWebSocketIngest:
    def __init__(self, url: str, on_event: Callable[[dict], Awaitable[None]]) -> None:
        self.url = url
        self.on_event = on_event
        self._stopped = False

    async def run(self) -> None:
        delay = 1
        while not self._stopped:
            try:
                async with websockets.connect(self.url) as websocket:
                    logger.info('Connected to STT websocket: %s', self.url)
                    delay = 1
                    async for message in websocket:
                        payload = json.loads(message)
                        event = normalize_event(payload)
                        await self.on_event(event.model_dump())
            except Exception as exc:
                logger.warning('STT websocket disconnected: %s', exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20)

    def stop(self) -> None:
        self._stopped = True
