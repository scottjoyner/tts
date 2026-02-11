from __future__ import annotations

import json
from collections.abc import Awaitable, Callable

from fastapi import WebSocket

from voicebus.schema.events import VoiceBusEvent

EventHandler = Callable[[VoiceBusEvent], Awaitable[None]]


class VoiceBusWSServer:
    def __init__(self) -> None:
        self.connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self.connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self.connections.discard(websocket)

    async def broadcast(self, event: VoiceBusEvent) -> None:
        encoded = event.model_dump_json()
        dead: list[WebSocket] = []
        for ws in self.connections:
            try:
                await ws.send_text(encoded)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)

    async def handle_ws(self, websocket: WebSocket, on_event: EventHandler | None = None) -> None:
        await self.connect(websocket)
        try:
            while True:
                payload = json.loads(await websocket.receive_text())
                if on_event:
                    await on_event(VoiceBusEvent(**payload))
        finally:
            self.disconnect(websocket)
