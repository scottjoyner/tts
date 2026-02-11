from __future__ import annotations


def evaluate_completion(iteration: int, max_iters: int, latest_result: str) -> tuple[bool, float]:
    confidence = min(1.0, 0.5 + (iteration / max_iters))
    normalized = latest_result.lower()
    complete = 'done=true' in normalized
    return complete, confidence
