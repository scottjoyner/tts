from __future__ import annotations

from pathlib import Path


class VoiceProfileManager:
    def __init__(self, root_dir: Path, default_voice: str) -> None:
        self.root_dir = root_dir
        self.default_voice = default_voice
        self.root_dir.mkdir(parents=True, exist_ok=True)

    def register(self, user_id: str) -> Path:
        profile_dir = self.root_dir / user_id
        profile_dir.mkdir(parents=True, exist_ok=True)
        return profile_dir

    def resolve_for_user(self, user_id: str | None) -> str:
        if user_id and (self.root_dir / user_id).exists():
            return user_id
        return self.default_voice
