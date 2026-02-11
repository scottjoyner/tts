from __future__ import annotations

import json
from pathlib import Path


class ModelRouter:
    def __init__(self, routing_path: Path) -> None:
        self.routing = json.loads(routing_path.read_text(encoding='utf-8'))

    def choose(self, use_case: str = 'planning') -> dict:
        candidates = [item for item in self.routing['routes'] if use_case in item.get('use_cases', [])]
        if not candidates:
            return self.routing['routes'][0]
        return sorted(candidates, key=lambda item: item.get('latency_weight', 1.0))[0]
