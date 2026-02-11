from __future__ import annotations

from fastapi import APIRouter, HTTPException

from tts_agent.tasks.store_sqlite import TaskStore


def build_tasks_router(store: TaskStore) -> APIRouter:
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

    return router
