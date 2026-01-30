from __future__ import annotations

import math
from pathlib import Path
from typing import Tuple

import numpy as np
import soundfile as sf


def read_audio(path: Path) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(path)
    if audio.ndim > 1:
        audio = np.mean(audio, axis=1)
    return audio.astype(np.float32), sr


def write_audio(path: Path, audio: np.ndarray, sr: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(path, audio, sr)


def duration_seconds(audio: np.ndarray, sr: int) -> float:
    return float(audio.shape[0] / sr)


def rms_db(audio: np.ndarray) -> float:
    if audio.size == 0:
        return -math.inf
    rms = np.sqrt(np.mean(np.square(audio)))
    if rms == 0:
        return -math.inf
    return 20 * math.log10(rms)


def clipping_percent(audio: np.ndarray, threshold: float = 0.99) -> float:
    if audio.size == 0:
        return 0.0
    clipped = np.sum(np.abs(audio) >= threshold)
    return float(clipped / audio.size * 100.0)
