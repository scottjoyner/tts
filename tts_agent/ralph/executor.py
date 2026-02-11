from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass
from typing import Any

from tts_agent.llm.router import StageModelRouter
from tts_agent.ralph.completion import completion_check
from tts_agent.ralph.context_builder import build_task_context
from tts_agent.ralph.iteration import IterationResult
from tts_agent.ralph.planner import make_plan
from tts_agent.ralph.scoring import score_iteration
from tts_agent.tasks.models import TaskRecord
from tts_agent.tasks.store_sqlite import TaskStore


@dataclass
class RalphResult:
    summary: str
    complete: bool
    iterations: int


class RalphExecutor:
    def __init__(
        self,
        store: TaskStore,
        max_iters: int,
        max_seconds: int,
        model_router: StageModelRouter | None = None,
        context_provider: Any | None = None,
    ) -> None:
        self.store = store
        self.max_iters = max_iters
        self.max_seconds = max_seconds
        self.model_router = model_router
        self.context_provider = context_provider

    async def run_task(self, task: TaskRecord) -> RalphResult:
        self.store.update_task_status(task.task_id, 'running')
        plan = make_plan(task.text)
        self.store.record_task_event(task.task_id, 'plan_created', json.dumps({'plan': plan}))

        start = time.monotonic()
        latest = IterationResult(
            iteration=0,
            plan=plan,
            next_action='start',
            tool_calls=[],
            proposed_response_text='Not started',
            completion_check={'done': False, 'reason': 'start'},
            confidence=0.0,
        )
        repeated_failures = 0
        for iteration in range(1, self.max_iters + 1):
            if time.monotonic() - start > self.max_seconds:
                break
            live = self.store.get_task(task.task_id)
            if live and live.status == 'cancelled':
                return RalphResult(summary='Task cancelled by user.', complete=False, iterations=iteration - 1)

            step = plan[(iteration - 1) % len(plan)]
            stage = 'execution'
            model = self.model_router.choose(stage)['model'] if self.model_router else 'rules-engine'
            context = build_task_context(self.context_provider, task) if self.context_provider else task.text
            text = f'Iteration {iteration}: step={step}. context={context[:80]}. done={iteration >= len(plan)}'
            done, reason = completion_check(task.text, text, iteration, self.max_iters)
            confidence = score_iteration(text, [])
            latest = IterationResult(
                iteration=iteration,
                plan=plan,
                next_action=step,
                tool_calls=[],
                proposed_response_text=text,
                completion_check={'done': done, 'reason': reason},
                confidence=confidence,
            )

            if confidence < 0.2:
                repeated_failures += 1
                if self.model_router:
                    self.model_router.mark_failure(model)
            else:
                repeated_failures = 0
                if self.model_router:
                    self.model_router.mark_success(model)

            self.store.record_llm_call(task.task_id, iteration, model, f'Step:{step}', text, stage=stage)
            self.store.record_task_event(task.task_id, 'ralph_iteration', latest.model_dump_json())
            if done:
                self.store.update_task_status(task.task_id, 'done', text)
                return RalphResult(summary=text, complete=True, iterations=iteration)

            if repeated_failures >= 2 and self.model_router:
                fallback = self.model_router.fallback_chain(stage)
                if fallback:
                    self.store.record_task_event(task.task_id, 'model_fallback', json.dumps({'to': fallback[0]['model']}))
            await asyncio.sleep(0)

        summary = f'Failed to complete within limits. Last result: {latest.proposed_response_text}'
        self.store.update_task_status(task.task_id, 'failed', summary)
        return RalphResult(summary=summary, complete=False, iterations=self.max_iters)

    async def run_tool(self, tool_name: str, payload: dict[str, Any]) -> dict[str, Any]:
        if tool_name == 'note_store':
            return {'ok': True, 'note': payload.get('note', '')}
        return {'ok': False, 'error': 'Tool disabled or unknown'}
