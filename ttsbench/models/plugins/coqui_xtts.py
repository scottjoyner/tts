from __future__ import annotations

import time
from importlib.util import find_spec
from pathlib import Path
from typing import Any, Dict

from ttsbench.models.base import BaseTTSModel, ModelCapabilities, SynthResult
from ttsbench.utils.audio import read_audio


class CoquiXTTSModel(BaseTTSModel):
    name = "coqui_xtts_v2"
    description = "Coqui XTTS v2 via TTS library"
    capabilities = ModelCapabilities(languages=["en", "es"], supports_cloning=True, supports_styles=True)

    @classmethod
    def is_available(cls) -> bool:
        return find_spec("TTS") is not None

    @classmethod
    def availability_help(cls) -> str:
        return (
            "Install Coqui TTS: pip install TTS. "
            "Ensure the XTTS v2 model is available locally; pass model_name or model_path."
        )

    def synth(self, text: str, config: Dict[str, Any], out_dir: Path) -> SynthResult:
        from TTS.api import TTS
        model_name = config.get("model_name", "tts_models/multilingual/multi-dataset/xtts_v2")
        model_path = config.get("model_path")
        speaker_wav = config.get("speaker_wav")
        language = config.get("language", "en")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "audio.wav"

        tts = TTS(model_name=model_name, model_path=model_path)
        start = time.perf_counter()
        tts.tts_to_file(
            text=text,
            file_path=str(output_path),
            speaker_wav=speaker_wav,
            language=language,
        )
        total = time.perf_counter() - start
        audio, sr = read_audio(output_path)
        duration = audio.shape[0] / sr if sr > 0 else 0.0
        timings = {
            "time_to_first_audio_ms": total * 1000.0,
            "total_time_s": total,
            "rtf": total / duration if duration > 0 else 0.0,
        }
        return SynthResult(audio_path=output_path, sample_rate=sr, timings=timings, stats={})
