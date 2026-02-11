from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tts_agent.tasks.models import TaskCreate
from tts_agent.tasks.queue import TaskQueue
from tts_agent.tasks.store_sqlite import TaskStore


def build_tasks_router(store: TaskStore, queue: TaskQueue | None = None) -> APIRouter:
    router = APIRouter(prefix='/tasks', tags=['tasks'])

    @router.post('/{task_id}/cancel')
    async def cancel_task(task_id: str) -> dict:
        if not store.cancel_task(task_id):
            raise HTTPException(status_code=404, detail='Task not cancellable')
        return {'task_id': task_id, 'status': 'cancelled'}

    @router.post('/cancel-active')
    async def cancel_active() -> dict:
        task = store.get_active_task()
        if not task:
            raise HTTPException(status_code=404, detail='No active task')
        store.cancel_task(task.task_id)
        return {'task_id': task.task_id, 'status': 'cancelled'}

    @router.get('')
    async def list_tasks(status: str | None = None) -> list[dict]:
        return [task.model_dump() for task in store.list_tasks(status=status)]

    @router.post('/enqueue')
    async def enqueue(payload: dict) -> dict:
        task = store.create_task(
            TaskCreate(
                source_segment_id=payload.get('source_segment_id', 'manual'),
                speaker_user=payload.get('speaker_user', 'default'),
                text=payload['text'],
                conversation_id=payload.get('conversation_id'),
                turn_id=payload.get('turn_id'),
                priority=payload.get('priority', 'interactive'),
            )
        )
        if queue is not None:
            await queue.put(task)
        return task.model_dump()

    return router
