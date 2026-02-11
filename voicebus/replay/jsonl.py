from __future__ import annotations

import json
from pathlib import Path
from typing import Iterable

from voicebus.schema.events import VoiceBusEvent


class JSONLReplay:
    def __init__(self, path: Path) -> None:
        self.path = path
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def append(self, event: VoiceBusEvent) -> None:
        with self.path.open('a', encoding='utf-8') as f:
            f.write(event.model_dump_json())
            f.write('\n')

    def read(self, cursor: int = 0) -> tuple[list[VoiceBusEvent], int]:
        events: list[VoiceBusEvent] = []
        with self.path.open('r', encoding='utf-8') as f:
            for index, line in enumerate(f):
                if index < cursor:
                    continue
                events.append(VoiceBusEvent(**json.loads(line)))
        return events, cursor + len(events)

    def write_many(self, events: Iterable[VoiceBusEvent]) -> int:
        count = 0
        for event in events:
            self.append(event)
            count += 1
        return count
