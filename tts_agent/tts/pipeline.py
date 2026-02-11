from __future__ import annotations

from collections.abc import AsyncIterator

from tts_agent.config import Settings
from tts_agent.tts.audio_output import LocalAudioOutput
from tts_agent.tts.chunker import chunk_text
from tts_agent.tts.engines.dummy_engine import DummyTTSEngine
from tts_agent.tts.engines.qwen3_tts_engine import Qwen3TTSEngine


class TTSPipeline:
    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.audio_output = LocalAudioOutput(enabled=settings.tts_play_local)
        if settings.tts_engine == 'qwen3':
            self.engine = Qwen3TTSEngine()
        else:
            self.engine = DummyTTSEngine()

    async def synthesize_stream(self, text: str, voice_profile: str | None = None) -> AsyncIterator[tuple[str, bytes]]:
        if hasattr(self.engine, 'synthesize_stream'):
            chunks = chunk_text(text, self.settings.tts_chunk_sentences)
            for chunk in chunks:
                async for audio in self.engine.synthesize_stream(
                    chunk,
                    sample_rate=self.settings.tts_sample_rate,
                    voice_profile=voice_profile,
                ):
                    self.audio_output.play(audio)
                    yield chunk, audio
            return

        for chunk in chunk_text(text, self.settings.tts_chunk_sentences):
            audio = await self.engine.synthesize(
                chunk,
                sample_rate=self.settings.tts_sample_rate,
                voice_profile=voice_profile,
            )
            self.audio_output.play(audio)
            yield chunk, audio
