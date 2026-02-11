from __future__ import annotations

import asyncio
from collections.abc import AsyncIterator
from pathlib import Path


class Qwen3TTSEngine:
    def __init__(self, model_path: Path | None = None) -> None:
        self.model_path = model_path

    async def synthesize(self, text: str, sample_rate: int = 24000, voice_profile: str | None = None) -> bytes:
        await asyncio.sleep(0)
        voice = voice_profile or 'default'
        return f'QWEN3_TTS[{sample_rate}|{voice}]: {text}'.encode('utf-8')

    async def synthesize_stream(
        self,
        text: str,
        sample_rate: int = 24000,
        voice_profile: str | None = None,
    ) -> AsyncIterator[bytes]:
        # optional streaming API surface compatible with qwen_tts style wrappers.
        sentences = [item.strip() for item in text.split('.') if item.strip()]
        if not sentences:
            return
        for sentence in sentences:
            yield await self.synthesize(sentence, sample_rate=sample_rate, voice_profile=voice_profile)
