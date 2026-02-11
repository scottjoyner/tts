from __future__ import annotations

import json
from pathlib import Path

from tts_agent.ingest.checkpoint_store import IngestCheckpointStore


class ReplayClient:
    def __init__(self, checkpoints: IngestCheckpointStore) -> None:
        self.checkpoints = checkpoints

    def ws_url_with_replay(self, base_url: str, source: str) -> str:
        event_id = self.checkpoints.last_event_id(source)
        if not event_id:
            return base_url
        join = '&' if '?' in base_url else '?'
        return f'{base_url}{join}cursor={event_id}'

    def iter_jsonl_from_offset(self, path: Path, source: str) -> tuple[list[dict], int]:
        offset = self.checkpoints.jsonl_offset(source)
        events: list[dict] = []
        with path.open('r', encoding='utf-8') as handle:
            handle.seek(offset)
            for line in handle:
                line = line.strip()
                if not line:
                    continue
                events.append(json.loads(line))
            new_offset = handle.tell()
        return events, new_offset
