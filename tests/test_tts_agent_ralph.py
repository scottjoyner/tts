import asyncio
from pathlib import Path

from tts_agent.ralph.executor import RalphExecutor
from tts_agent.tasks.models import TaskCreate
from tts_agent.tasks.store_sqlite import TaskStore


def test_ralph_respects_stop_conditions(tmp_path: Path) -> None:
    store = TaskStore(tmp_path / 'tasks.db')
    task = store.create_task(
        TaskCreate(source_segment_id='seg-2', speaker_user='primary_user', text='do complex thing')
    )
    executor = RalphExecutor(store=store, max_iters=1, max_seconds=1)
    result = asyncio.run(executor.run_task(task))

    assert result.iterations == 1
    assert result.complete is False
    loaded = store.get_task(task.task_id)
    assert loaded is not None
    assert loaded.status == 'failed'
