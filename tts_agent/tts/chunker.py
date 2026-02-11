from __future__ import annotations

import re


def chunk_text(text: str, sentences_per_chunk: int) -> list[str]:
    sentences = [item.strip() for item in re.split(r'(?<=[.!?])\\s+', text) if item.strip()]
    if not sentences:
        return []
    return [
        ' '.join(sentences[idx : idx + sentences_per_chunk])
        for idx in range(0, len(sentences), sentences_per_chunk)
    ]
