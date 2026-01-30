from datetime import datetime
from importlib.util import find_spec
from pathlib import Path

import pytest
import sqlite3

from ttsbench.utils.results import ResultsWriter, RunInfo


def test_results_writer(tmp_path: Path) -> None:
    if find_spec("sqlalchemy") is None:
        pytest.skip("sqlalchemy not available")
    sqlite_path = tmp_path / "results.sqlite"
    writer = ResultsWriter(sqlite_path)
    run_info = RunInfo(run_id="run1", created_at=datetime.utcnow(), prompts_path="prompts.yaml", seed=42)
    writer.write_run(run_info)

    model_ids = writer.write_models(
        "run1", [{"name": "piper", "description": "Piper", "available": True}]
    )
    prompt_ids = writer.write_prompts(
        "run1",
        [{"id": "p1", "text": "hello", "language": "en", "style": "neutral"}],
    )
    writer.write_output(
        run_id="run1",
        model_id=model_ids[0],
        prompt_id=prompt_ids[0],
        audio_path="audio.wav",
        sample_rate=24000,
        metrics={"duration_s": 1.2},
    )

    conn = sqlite3.connect(sqlite_path)
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM outputs")
    assert cursor.fetchone()[0] == 1
    conn.close()
