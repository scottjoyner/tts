from __future__ import annotations


class DummyTTSEngine:
    async def synthesize(self, text: str, sample_rate: int = 24000) -> bytes:
        payload = f'DUMMY_TTS[{sample_rate}]: {text}'.encode('utf-8')
        return payload
