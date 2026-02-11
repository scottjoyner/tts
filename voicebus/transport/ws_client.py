from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator

import websockets

from voicebus.schema.events import VoiceBusEvent


class VoiceBusWSClient:
    def __init__(self, url: str) -> None:
        self.url = url

    async def subscribe(self) -> AsyncIterator[VoiceBusEvent]:
        async with websockets.connect(self.url) as ws:
            while True:
                raw = await ws.recv()
                payload = json.loads(raw)
                yield VoiceBusEvent(**payload)

    async def publish(self, events: list[VoiceBusEvent]) -> None:
        async with websockets.connect(self.url) as ws:
            for event in events:
                await ws.send(event.model_dump_json())
                await asyncio.sleep(0)
