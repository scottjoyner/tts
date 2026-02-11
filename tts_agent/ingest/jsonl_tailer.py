from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from tts_agent.ingest.checkpoint_store import IngestCheckpointStore
from tts_agent.ingest.normalizer import normalize_event
from tts_agent.ingest.replay_client import ReplayClient


class JSONLTailer:
    def __init__(
        self,
        root_dir: Path,
        on_event: Callable[[dict], Awaitable[None]],
        checkpoints: IngestCheckpointStore | None = None,
    ) -> None:
        self.root_dir = root_dir
        self.on_event = on_event
        self.checkpoints = checkpoints
        self.replay = ReplayClient(checkpoints) if checkpoints else None

    async def run(self) -> None:
        seen_offsets: dict[Path, int] = {}
        while True:
            for path in sorted(self.root_dir.glob('events/**/session_*.jsonl')):
                if not path.exists():
                    continue
                source = f'jsonl:{path}'
                if self.replay:
                    events, new_offset = self.replay.iter_jsonl_from_offset(path, source)
                    for payload in events:
                        raw = json.dumps(payload, sort_keys=True)
                        if self.checkpoints and self.checkpoints.seen_event(raw):
                            continue
                        event = normalize_event(payload)
                        await self.on_event(event.model_dump())
                        self.checkpoints.update_checkpoint(source, event_id=event.event_id or event.segment_id)
                    self.checkpoints.update_checkpoint(source, offset=new_offset)
                else:
                    offset = seen_offsets.get(path, 0)
                    with path.open('r', encoding='utf-8') as handle:
                        handle.seek(offset)
                        for line in handle:
                            if not line.strip():
                                continue
                            payload = json.loads(line)
                            event = normalize_event(payload)
                            await self.on_event(event.model_dump())
                        seen_offsets[path] = handle.tell()
            await asyncio.sleep(0.5)
