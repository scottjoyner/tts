from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelHealth:
    failures: int = 0
    successes: int = 0

    @property
    def score(self) -> float:
        total = self.failures + self.successes
        if total == 0:
            return 1.0
        return self.successes / total


class StageModelRouter:
    def __init__(self, routing_path: Path) -> None:
        payload = json.loads(routing_path.read_text(encoding='utf-8'))
        self.routes = payload.get('routes', [])
        self.health: dict[str, ModelHealth] = {}

    def choose(self, stage: str) -> dict:
        candidates = [route for route in self.routes if stage in route.get('use_cases', [])]
        if not candidates:
            candidates = self.routes
        ranked = sorted(candidates, key=lambda r: (self.health.get(r['model'], ModelHealth()).score, -r.get('quality_weight', 1)), reverse=True)
        return ranked[0]

    def fallback_chain(self, stage: str) -> list[dict]:
        candidates = [route for route in self.routes if stage in route.get('use_cases', [])]
        if not candidates:
            return self.routes
        return sorted(candidates, key=lambda r: r.get('latency_weight', 1.0))

    def mark_success(self, model: str) -> None:
        health = self.health.setdefault(model, ModelHealth())
        health.successes += 1

    def mark_failure(self, model: str) -> None:
        health = self.health.setdefault(model, ModelHealth())
        health.failures += 1

    def snapshot(self) -> dict[str, float]:
        return {model: stats.score for model, stats in self.health.items()}
