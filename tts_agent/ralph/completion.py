from __future__ import annotations

import re


_COMPLETION_PATTERNS = (r'\bcompleted\b', r'\bfinished\b', r'\bdone\b(?!\s*=\s*false)')


def completion_check(task_text: str, response_text: str, iteration: int, max_iters: int) -> tuple[bool, str]:
    lowered = response_text.lower()
    if any(re.search(pattern, lowered) for pattern in _COMPLETION_PATTERNS):
        return True, 'response_signals_completion'
    if 'ask user' in lowered:
        return False, 'awaiting_user_input'
    if len(task_text) < 6 and iteration >= 1:
        return True, 'short_task_rule'
    return False, 'needs_more_work'
