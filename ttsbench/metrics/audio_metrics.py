from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from ttsbench.utils.audio import clipping_percent, duration_seconds, read_audio, rms_db


class AudioMetrics:
    def __init__(self, audio_path: Path) -> None:
        self.audio_path = audio_path
        self.audio, self.sr = read_audio(audio_path)

    def compute(self) -> Dict[str, float]:
        metrics = {
            "duration_s": duration_seconds(self.audio, self.sr),
            "rms_db": rms_db(self.audio),
            "clipping_pct": clipping_percent(self.audio),
        }
        lufs = estimate_lufs(self.audio_path)
        if lufs is not None:
            metrics["lufs"] = lufs
        return metrics


def estimate_lufs(audio_path: Path) -> Optional[float]:
    try:
        result = subprocess.run(
            [
                "ffmpeg",
                "-hide_banner",
                "-nostats",
                "-i",
                str(audio_path),
                "-filter_complex",
                "ebur128=framelog=verbose",
                "-f",
                "null",
                "-",
            ],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=False,
            text=True,
        )
    except FileNotFoundError:
        return None
    for line in result.stderr.splitlines():
        if "I:" in line and "LUFS" in line:
            parts = line.split("I:")
            if len(parts) < 2:
                continue
            try:
                lufs = float(parts[1].split("LUFS")[0].strip())
                return lufs
            except ValueError:
                continue
    return None
