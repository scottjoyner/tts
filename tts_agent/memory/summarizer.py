from __future__ import annotations

from tts_agent.memory.conversation_store import ConversationStore


class ConversationSummarizer:
    def __init__(self, store: ConversationStore, every_n_turns: int = 4) -> None:
        self.store = store
        self.every_n_turns = every_n_turns

    def maybe_rollup(self, conversation_id: str) -> str:
        turns = self.store.recent_turns(conversation_id)
        if not turns:
            return ''
        if len(turns) % self.every_n_turns != 0:
            return self.store.get_summary(conversation_id)
        condensed = ' | '.join(f"{t['role']}: {t['text'][:80]}" for t in turns[-self.every_n_turns :])
        previous = self.store.get_summary(conversation_id)
        summary = (previous + ' ' + condensed).strip()[:600]
        self.store.upsert_summary(conversation_id, summary)
        return summary
