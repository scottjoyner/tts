from __future__ import annotations

import asyncio
from typing import Any

from fastapi import WebSocket


class WSConnectionHub:
    def __init__(self) -> None:
        self._connections: set[WebSocket] = set()

    async def connect(self, websocket: WebSocket) -> None:
        await websocket.accept()
        self._connections.add(websocket)

    def disconnect(self, websocket: WebSocket) -> None:
        self._connections.discard(websocket)

    async def broadcast_json(self, payload: dict[str, Any]) -> None:
        await self._broadcast(lambda ws: ws.send_json(payload))

    async def broadcast_bytes(self, payload: bytes) -> None:
        await self._broadcast(lambda ws: ws.send_bytes(payload))

    async def _broadcast(self, sender: Any) -> None:
        dead: list[WebSocket] = []
        for websocket in self._connections:
            try:
                await sender(websocket)
            except Exception:
                dead.append(websocket)
        for websocket in dead:
            self.disconnect(websocket)
        await asyncio.sleep(0)
