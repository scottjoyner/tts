from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable
from pathlib import Path

from tts_agent.ingest.normalizer import normalize_event


class JSONLTailer:
    def __init__(self, root_dir: Path, on_event: Callable[[dict], Awaitable[None]]) -> None:
        self.root_dir = root_dir
        self.on_event = on_event

    async def run(self) -> None:
        seen_offsets: dict[Path, int] = {}
        while True:
            for path in sorted(self.root_dir.glob('events/**/session_*.jsonl')):
                offset = seen_offsets.get(path, 0)
                if not path.exists():
                    continue
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
