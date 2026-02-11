from __future__ import annotations

import difflib
import subprocess
from pathlib import Path


class WorkspaceTools:
    def __init__(self, workspace: Path, enable_tests: bool = False) -> None:
        self.workspace = workspace
        self.enable_tests = enable_tests

    def read_file(self, relpath: str) -> str:
        return (self.workspace / relpath).read_text(encoding='utf-8')

    def write_file(self, relpath: str, content: str) -> None:
        path = self.workspace / relpath
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding='utf-8')

    def apply_patch(self, relpath: str, new_content: str) -> str:
        old = ''
        path = self.workspace / relpath
        if path.exists():
            old = path.read_text(encoding='utf-8')
        self.write_file(relpath, new_content)
        return ''.join(
            difflib.unified_diff(
                old.splitlines(keepends=True),
                new_content.splitlines(keepends=True),
                fromfile=f'a/{relpath}',
                tofile=f'b/{relpath}',
            )
        )

    def run_tests(self, cmd: str = 'pytest -q') -> tuple[bool, str]:
        if not self.enable_tests:
            return False, 'tests disabled by config'
        completed = subprocess.run(cmd, shell=True, cwd=self.workspace, capture_output=True, text=True)
        return completed.returncode == 0, (completed.stdout + '\n' + completed.stderr).strip()
