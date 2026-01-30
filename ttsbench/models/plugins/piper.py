from __future__ import annotations

import subprocess
import time
from pathlib import Path
from typing import Any, Dict

from ttsbench.models.base import BaseTTSModel, ModelCapabilities, SynthResult
from ttsbench.utils.audio import read_audio


class PiperModel(BaseTTSModel):
    name = "piper"
    description = "Piper local CLI"
    capabilities = ModelCapabilities(languages=["en", "es"], supports_cloning=False, supports_styles=False)

    @classmethod
    def is_available(cls) -> bool:
        return _which("piper") is not None

    @classmethod
    def availability_help(cls) -> str:
        return "Install piper-tts and ensure `piper` is on PATH. Provide a .onnx voice file."

    def synth(self, text: str, config: Dict[str, Any], out_dir: Path) -> SynthResult:
        voice = config.get("voice")
        if not voice:
            raise ValueError("Piper requires config['voice'] pointing to a .onnx voice file.")
        out_dir.mkdir(parents=True, exist_ok=True)
        output_path = out_dir / "audio.wav"
        cmd = [
            "piper",
            "--model",
            str(voice),
            "--output_file",
            str(output_path),
        ]
        if config.get("speaker"):
            cmd += ["--speaker", str(config["speaker"])]
        start = time.perf_counter()
        subprocess.run(cmd, input=text, text=True, check=True)
        total = time.perf_counter() - start
        audio, sr = read_audio(output_path)
        duration = audio.shape[0] / sr if sr > 0 else 0.0
        timings = {
            "time_to_first_audio_ms": total * 1000.0,
            "total_time_s": total,
            "rtf": total / duration if duration > 0 else 0.0,
        }
        stats: Dict[str, float] = {}
        return SynthResult(audio_path=output_path, sample_rate=sr, timings=timings, stats=stats)


def _which(binary: str) -> str | None:
    try:
        result = subprocess.run(["which", binary], capture_output=True, text=True, check=False)
    except Exception:
        return None
    path = result.stdout.strip()
    return path or None
