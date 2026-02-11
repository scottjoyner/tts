from __future__ import annotations

from tts_agent.memory.conversation_store import ConversationStore
from tts_agent.tasks.models import TaskRecord


def build_task_context(store: ConversationStore, task: TaskRecord) -> str:
    if not task.conversation_id:
        return f'Current task: {task.text}'
    turns = store.recent_turns(task.conversation_id)
    summary = store.get_summary(task.conversation_id)
    rendered_turns = '\n'.join(f"- {turn['role']}: {turn['text']}" for turn in turns)
    return (
        f'Conversation summary: {summary or "(none)"}\n'
        f'Last turns:\n{rendered_turns or "(none)"}\n'
        f'Current task: {task.text}'
    )
