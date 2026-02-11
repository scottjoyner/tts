from __future__ import annotations

from tts_agent.config import Settings
from tts_agent.tasks.models import SegmentEvent


class SegmentRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_authenticated(self, event: SegmentEvent) -> bool:
        user = event.speaker.authenticated_user or event.speaker.candidate
        if not event.speaker.authenticated and event.speaker.authenticated_user is None:
            return False
        if self.settings.allowed_users and user and user not in self.settings.allowed_users:
            return False
        if self.settings.auth_strict and (event.speaker.score or 0) < self.settings.auth_score_min:
            return False
        return event.speaker.authenticated or event.speaker.authenticated_user is not None

    def is_actionable(self, event: SegmentEvent) -> bool:
        if event.actionability.actionable:
            return True
        text = (event.transcript_final or event.text or '').strip().lower()
        if not text:
            return False
        if event.trigger_context.triggered:
            return True
        rule_prefixes = ('please', 'could you', 'can you', 'start', 'stop', 'summarize', 'tell me', 'fix', 'write code')
        return any(text.startswith(prefix) for prefix in rule_prefixes)
