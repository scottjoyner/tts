from __future__ import annotations

from tts_agent.config import Settings
from tts_agent.tasks.models import SegmentEvent


class SegmentRouter:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings

    def is_authenticated(self, event: SegmentEvent) -> bool:
        user = event.authenticated_user or event.speaker_candidate
        if event.authenticated is False and event.authenticated_user is None:
            return False
        if self.settings.allowed_users and user and user not in self.settings.allowed_users:
            return False
        if self.settings.auth_strict and (event.speaker_score or 0) < self.settings.auth_score_min:
            return False
        return event.authenticated or event.authenticated_user is not None

    def is_actionable(self, event: SegmentEvent) -> bool:
        if bool(event.actionability.get('is_actionable')):
            return True
        text = (event.transcript_final or '').strip().lower()
        if not text:
            return False
        if event.is_command or event.triggered:
            return True
        rule_prefixes = ('please', 'could you', 'can you', 'start', 'stop', 'summarize', 'tell me')
        return any(text.startswith(prefix) for prefix in rule_prefixes)
