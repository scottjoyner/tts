from importlib.util import find_spec
from pathlib import Path

import pytest

from ttsbench.metrics.audio_metrics import AudioMetrics
from ttsbench.utils.audio import write_audio


def test_audio_metrics(tmp_path: Path) -> None:
    if find_spec("numpy") is None or find_spec("soundfile") is None:
        pytest.skip("numpy/soundfile not available")
    import numpy as np

    sr = 16000
    t = np.linspace(0, 1.0, sr, endpoint=False)
    audio = 0.1 * np.sin(2 * np.pi * 440 * t)
    audio_path = tmp_path / "tone.wav"
    write_audio(audio_path, audio, sr)

    metrics = AudioMetrics(audio_path).compute()
    assert 0.9 < metrics["duration_s"] < 1.1
    assert metrics["clipping_pct"] == 0.0
