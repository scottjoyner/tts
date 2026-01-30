from __future__ import annotations

from dataclasses import dataclass
from importlib.util import find_spec
from pathlib import Path
from typing import Dict, Optional

from jiwer import cer, wer


@dataclass
class ASRResult:
    transcript: str
    wer: float
    cer: float


def run_asr(audio_path: Path, language: str = "en") -> Optional[str]:
    if find_spec("faster_whisper") is None:
        return None
    from faster_whisper import WhisperModel
    model = WhisperModel("small", device="cpu", compute_type="int8")
    segments, _info = model.transcribe(str(audio_path), language=language)
    transcript = " ".join(segment.text for segment in segments).strip()
    return transcript


def compute_asr_metrics(audio_path: Path, reference: str, language: str = "en") -> Optional[Dict[str, float]]:
    transcript = run_asr(audio_path, language)
    if transcript is None:
        return None
    normalized_ref = reference.strip().lower()
    normalized_hyp = transcript.strip().lower()
    return {
        "wer": float(wer(normalized_ref, normalized_hyp)),
        "cer": float(cer(normalized_ref, normalized_hyp)),
    }
