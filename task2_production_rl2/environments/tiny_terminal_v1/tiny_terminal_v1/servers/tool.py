from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field


class TerminalState(vf.State):
    files: dict[str, str] = Field(default_factory=dict)
    submitted: bool = False
    score: float = 0.0


class TinyTerminalToolset(vf.Toolset[vf.ToolsetConfig, TerminalState]):
    TOOL_PREFIX = "terminal"

    async def setup_task(self, task) -> None:
        self._expected = dict(task.expected_files)
        self.state.files = dict(task.initial_files)

    @vf.tool
    def list_files(self) -> list[str]:
        """List the files in the controlled terminal workspace."""
        return sorted(self.state.files)

    @vf.tool
    def read(self, path: str) -> str:
        """Read a controlled workspace file."""
        return self.state.files[path]

    @vf.tool
    def write(self, path: str, content: str) -> str:
        """Write a controlled workspace file."""
        if "/" in path or path in {".", ".."}:
            raise ValueError("only top-level controlled files are allowed")
        self.state.files[path] = content
        return "ok"

    @vf.tool
    def remove(self, path: str) -> str:
        """Remove a controlled workspace file."""
        self.state.files.pop(path)
        return "ok"

    @vf.tool
    def submit(self) -> dict[str, object]:
        """Score the objective final filesystem state."""
        self.state.submitted = True
        self.state.score = float(self.state.files == self._expected)
        return {"passed": bool(self.state.score)}


if __name__ == "__main__":
    TinyTerminalToolset.run()
