from __future__ import annotations

from dataclasses import dataclass


@dataclass
class ModuleResult:
    text: str


class _BaseModule:
    def __call__(self, prompt: str) -> ModuleResult:
        return ModuleResult(text=prompt)


class IssueUnderstanding(_BaseModule):
    pass


class PlanSynthesis(_BaseModule):
    pass


class PatchProposal(_BaseModule):
    pass


class PatchApplication(_BaseModule):
    pass


class TestGeneration(_BaseModule):
    pass


class PatchEvaluation(_BaseModule):
    pass


def maybe_compile() -> str:
    try:
        import dspy  # type: ignore

        return f'dspy_available:{getattr(dspy, "__version__", "unknown")}'
    except Exception:
        return 'dspy_unavailable'
