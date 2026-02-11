from __future__ import annotations

import asyncio
import json
import logging
from collections.abc import Awaitable, Callable

import websockets

from tts_agent.ingest.checkpoint_store import IngestCheckpointStore
from tts_agent.ingest.normalizer import normalize_event
from tts_agent.ingest.replay_client import ReplayClient

logger = logging.getLogger(__name__)


class STTWebSocketIngest:
    def __init__(
        self,
        url: str,
        on_event: Callable[[dict], Awaitable[None]],
        checkpoints: IngestCheckpointStore | None = None,
        source: str = 'stt_ws',
    ) -> None:
        self.url = url
        self.on_event = on_event
        self._stopped = False
        self.checkpoints = checkpoints
        self.source = source
        self.replay_client = ReplayClient(checkpoints) if checkpoints else None

    async def run(self) -> None:
        delay = 1
        while not self._stopped:
            try:
                ws_url = self.replay_client.ws_url_with_replay(self.url, self.source) if self.replay_client else self.url
                async with websockets.connect(ws_url) as websocket:
                    logger.info('Connected to STT websocket: %s', ws_url)
                    delay = 1
                    async for message in websocket:
                        if self.checkpoints and self.checkpoints.seen_event(message):
                            continue
                        payload = json.loads(message)
                        event = normalize_event(payload)
                        event_data = event.model_dump()
                        await self.on_event(event_data)
                        if self.checkpoints:
                            self.checkpoints.update_checkpoint(
                                self.source,
                                event_id=event_data.get('event_id') or event_data.get('segment_id'),
                            )
            except Exception as exc:
                logger.warning('STT websocket disconnected: %s', exc)
                await asyncio.sleep(delay)
                delay = min(delay * 2, 20)

    def stop(self) -> None:
        self._stopped = True
