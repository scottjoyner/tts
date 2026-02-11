from __future__ import annotations


def score_iteration(response_text: str, tool_calls: list[dict]) -> float:
    score = 0.35
    if len(response_text.split()) > 8:
        score += 0.2
    if 'error' in response_text.lower():
        score -= 0.25
    if tool_calls:
        score += 0.15
    return max(0.0, min(1.0, score))
