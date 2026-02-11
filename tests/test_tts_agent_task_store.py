from pathlib import Path

from tts_agent.tasks.models import TaskCreate
from tts_agent.tasks.store_sqlite import TaskStore


def test_task_store_persistence(tmp_path: Path) -> None:
    db_path = tmp_path / 'tasks.db'
    store = TaskStore(db_path)
    task = store.create_task(
        TaskCreate(source_segment_id='seg-1', speaker_user='primary_user', text='do thing')
    )

    loaded = store.get_task(task.task_id)
    assert loaded is not None
    assert loaded.text == 'do thing'
    store.update_task_status(task.task_id, 'done', 'completed')
    updated = store.get_task(task.task_id)
    assert updated is not None
    assert updated.status == 'done'
    assert updated.result_summary == 'completed'
