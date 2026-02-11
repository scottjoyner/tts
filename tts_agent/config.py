from __future__ import annotations

from pathlib import Path
from typing import Literal

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8', extra='ignore')

    host: str = '0.0.0.0'
    port: int = 8010

    stt_ws_url: str = 'ws://127.0.0.1:8000/ws/events'
    stt_jsonl_dir: Path = Path('data/stt_events')
    stt_ingest_mode: Literal['ws', 'tail', 'both'] = 'ws'

    auth_strict: bool = True
    auth_score_min: float = 0.72
    auth_allowed_users: str = 'primary_user'

    task_db_path: Path = Path('data/tasks.db')
    events_jsonl_path: Path = Path('data/logs/events.jsonl')
    artifacts_dir: Path = Path('data/artifacts')

    ralph_max_iters: int = 12
    ralph_max_seconds: int = 180
    ralph_model_routing_path: Path = Path('routing.json')
    ralph_progress_speech: bool = True

    task_parallelism: int = 2
    conversation_memory_turns: int = 10
    conversation_summary_every: int = 4

    tts_engine: Literal['qwen3', 'dummy'] = 'dummy'
    tts_voice_profile_dir: Path = Path('data/voices')
    tts_default_voice: str = 'primary_user'
    tts_stream_format: Literal['pcm16'] = 'pcm16'
    tts_sample_rate: int = 24_000
    tts_chunk_sentences: int = Field(default=2, ge=1, le=3)
    tts_play_local: bool = False

    @property
    def allowed_users(self) -> set[str]:
        return {item.strip() for item in self.auth_allowed_users.split(',') if item.strip()}


settings = Settings()
