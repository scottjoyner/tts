from __future__ import annotations

PRIORITY_RANK = {'critical': 0, 'interactive': 1, 'background': 2}


def rank(priority: str) -> int:
    return PRIORITY_RANK.get(priority, 1)
