from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
from typing import Optional

import numpy as np


def _load_embedding(path: Path) -> Optional[np.ndarray]:
    if find_spec("resemblyzer") is None:
        return None
    from resemblyzer import VoiceEncoder, preprocess_wav

    wav = preprocess_wav(str(path))
    encoder = VoiceEncoder()
    emb = encoder.embed_utterance(wav)
    return np.asarray(emb)


def cosine_similarity(ref_path: Path, sample_path: Path) -> Optional[float]:
    ref_emb = _load_embedding(ref_path)
    sample_emb = _load_embedding(sample_path)
    if ref_emb is None or sample_emb is None:
        return None
    numerator = float(np.dot(ref_emb, sample_emb))
    denominator = float(np.linalg.norm(ref_emb) * np.linalg.norm(sample_emb))
    if denominator == 0:
        return None
    return numerator / denominator
