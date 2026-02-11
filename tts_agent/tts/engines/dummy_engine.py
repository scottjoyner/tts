from __future__ import annotations


class DummyTTSEngine:
    async def synthesize(self, text: str, sample_rate: int = 24000, voice_profile: str | None = None) -> bytes:
        voice = voice_profile or 'default'
        payload = f'DUMMY_TTS[{sample_rate}|{voice}]: {text}'.encode('utf-8')
        return payload
