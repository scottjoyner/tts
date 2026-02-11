from __future__ import annotations

import hashlib


def make_signature(turn_id: str, text: str) -> str:
    normalized = ' '.join(text.strip().lower().split())
    return hashlib.sha256(f'{turn_id}:{normalized}'.encode('utf-8')).hexdigest()
