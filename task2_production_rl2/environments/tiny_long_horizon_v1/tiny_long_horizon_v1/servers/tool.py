from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field


class LongState(vf.State):
    completed: list[str] = Field(default_factory=list)
    score: float = 0.0


class TinyLongHorizonToolset(vf.Toolset[vf.ToolsetConfig, LongState]):
    TOOL_PREFIX = "workflow"

    async def setup_task(self, task) -> None:
        self._steps = list(task.steps)

    @vf.tool
    def next_step(self) -> str | None:
        """Return the next required workflow step."""
        index = len(self.state.completed)
        return self._steps[index] if index < len(self._steps) else None

    @vf.tool
    def complete(self, step: str) -> str:
        """Complete exactly the next workflow step."""
        expected = self._steps[len(self.state.completed)]
        if step != expected:
            raise ValueError(f"expected {expected!r}")
        self.state.completed.append(step)
        return "ok"

    @vf.tool
    def submit(self) -> dict[str, object]:
        """Submit only after the ordered workflow is complete."""
        self.state.score = float(self.state.completed == self._steps)
        return {"passed": bool(self.state.score)}


if __name__ == "__main__":
    TinyLongHorizonToolset.run()
