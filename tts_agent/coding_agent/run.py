from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from tts_agent.coding_agent.dspy_modules import (
    IssueUnderstanding,
    PatchApplication,
    PatchEvaluation,
    PatchProposal,
    PlanSynthesis,
    TestGeneration,
    maybe_compile,
)
from tts_agent.coding_agent.tools import WorkspaceTools
from tts_agent.tasks.models import TaskRecord


@dataclass
class CodingResult:
    changed_files: list[str]
    patch_diff: str
    tests_executed: list[str]
    test_results: list[str]
    evaluation_score: float
    summary: str


def run_coding_task(task: TaskRecord, workspace: Path) -> CodingResult:
    tools = WorkspaceTools(workspace)
    compile_status = maybe_compile()

    issue = IssueUnderstanding()(task.text).text
    plan = PlanSynthesis()(f'Issue: {issue}').text
    proposal = PatchProposal()(f'Plan: {plan}').text

    target = 'generated/agent_patch.txt'
    content = (
        f'# Generated patch intent\n'
        f'task_id={task.task_id}\n'
        f'compile={compile_status}\n'
        f'issue={issue}\n'
        f'plan={plan}\n'
        f'proposal={proposal}\n'
    )
    patch_diff = tools.apply_patch(target, PatchApplication()(content).text)

    tests = [TestGeneration()('pytest -q').text]
    ok, output = tools.run_tests(tests[0])
    eval_text = PatchEvaluation()(output or 'no tests executed').text
    score = 0.8 if ok else 0.55

    return CodingResult(
        changed_files=[target],
        patch_diff=patch_diff,
        tests_executed=tests,
        test_results=[output],
        evaluation_score=score,
        summary=f'Coding task applied ({compile_status}): {eval_text[:120]}',
    )
