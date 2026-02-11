from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from tts_agent.ralph.evaluator import evaluate_completion
from tts_agent.ralph.planner import make_plan
from tts_agent.tasks.models import TaskRecord
from tts_agent.tasks.store_sqlite import TaskStore


@dataclass
class RalphResult:
    summary: str
    complete: bool
    iterations: int


class RalphExecutor:
    def __init__(self, store: TaskStore, max_iters: int, max_seconds: int) -> None:
        self.store = store
        self.max_iters = max_iters
        self.max_seconds = max_seconds

    async def run_task(self, task: TaskRecord) -> RalphResult:
        self.store.update_task_status(task.task_id, 'running')
        plan = make_plan(task.text)
        self.store.record_task_event(task.task_id, 'plan_created', json.dumps({'plan': plan}))

        start = time.monotonic()
        latest_result = 'Not started'
        for iteration in range(1, self.max_iters + 1):
            if time.monotonic() - start > self.max_seconds:
                break
            step = plan[(iteration - 1) % len(plan)]
            latest_result = f'Iteration {iteration}: finished step: {step}. done={iteration >= len(plan)}'
            self.store.record_llm_call(
                task.task_id,
                iteration,
                'rules-engine',
                f'Step: {step}',
                latest_result,
            )
            complete, confidence = evaluate_completion(iteration, self.max_iters, latest_result)
            self.store.record_task_event(
                task.task_id,
                'iteration',
                json.dumps({'iteration': iteration, 'result': latest_result, 'confidence': confidence}),
            )
            if complete:
                self.store.update_task_status(task.task_id, 'done', latest_result)
                return RalphResult(summary=latest_result, complete=True, iterations=iteration)
            await asyncio.sleep(0)

        summary = f'Failed to complete within limits. Last result: {latest_result}'
        self.store.update_task_status(task.task_id, 'failed', summary)
        return RalphResult(summary=summary, complete=False, iterations=self.max_iters)

    async def run_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name == 'note_store':
            return {'ok': True, 'note': payload.get('note', '')}
        return {'ok': False, 'error': 'Tool disabled or unknown'}
