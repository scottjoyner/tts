from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from tts_agent.utils.time import utc_now_iso


class JSONLWriter:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def write(self, event_type: str, payload: dict[str, Any]) -> None:
        record = {
            'ts': utc_now_iso(),
            'event_type': event_type,
            'payload': payload,
        }
        with self.path.open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(record, ensure_ascii=False) + '\n')
