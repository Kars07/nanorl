from __future__ import annotations

import verifiers.v1 as vf


class TooluseState(vf.State):
    queries: int = 0
    score: float = 0.0


class TinyTooluseToolset(vf.Toolset[vf.ToolsetConfig, TooluseState]):
    TOOL_PREFIX = "issues"

    async def setup_task(self, task) -> None:
        self._records = dict(task.records)
        self._expected = task.expected

    @vf.tool
    def query(self, issue_id: str) -> dict[str, str]:
        """Retrieve one issue record from the task database."""
        self.state.queries += 1
        return self._records[issue_id]

    @vf.tool
    def submit(self, answer: str) -> dict[str, object]:
        """Submit the derived database answer."""
        self.state.score = float(answer.strip().casefold() == self._expected.casefold())
        return {"passed": bool(self.state.score)}


if __name__ == "__main__":
    TinyTooluseToolset.run()
