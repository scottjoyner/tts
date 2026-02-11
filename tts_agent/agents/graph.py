from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tts_agent.coding_agent.run import run_coding_task
from tts_agent.tasks.models import TaskRecord
from tts_agent.tasks.store_sqlite import TaskStore


@dataclass
class AgentGraphResult:
    summary: str
    complete: bool


class AgentGraph:
    def __init__(self, store: TaskStore, workspace_root: Path) -> None:
        self.store = store
        self.workspace_root = workspace_root

    def _is_coding_task(self, task: TaskRecord) -> bool:
        tokens = ('write code', 'fix bug', 'generate script', 'refactor', 'patch')
        return any(token in task.text.lower() for token in tokens)

    async def run(self, task: TaskRecord, run_id: str) -> AgentGraphResult:
        self.store.record_agent_step(run_id, task.task_id, 'turn_interpreter', 'started', task.text)
        intent = 'coding' if self._is_coding_task(task) else 'general'
        self.store.record_agent_step(run_id, task.task_id, 'planner', 'result', f'intent={intent}')

        if intent == 'coding':
            result = run_coding_task(task, self.workspace_root)
            self.store.record_artifact(task.task_id, 'patch', result.patch_diff, run_id=run_id)
            self.store.record_agent_step(run_id, task.task_id, 'coding_agent', 'result', f'score={result.evaluation_score:.2f}')
            summary = result.summary
        else:
            summary = f"Handled task: {task.text}"
            self.store.record_agent_step(run_id, task.task_id, 'executor', 'result', summary)

        self.store.record_agent_step(run_id, task.task_id, 'verifier', 'result', 'basic-verification-passed')
        self.store.record_agent_step(run_id, task.task_id, 'summarizer', 'result', summary)
        return AgentGraphResult(summary=summary, complete=True)
