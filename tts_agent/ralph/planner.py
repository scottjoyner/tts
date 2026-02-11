from __future__ import annotations

from tts_agent.ralph.prompts import PLAN_PROMPT


def make_plan(task_text: str) -> list[str]:
    _ = PLAN_PROMPT.format(task_text=task_text)
    base_steps = [
        'Understand the request and constraints',
        'Execute the required action using available tools',
        'Summarize the outcome and next steps',
    ]
    return base_steps
