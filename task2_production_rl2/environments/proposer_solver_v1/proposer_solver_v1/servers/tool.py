from __future__ import annotations

import verifiers.v1 as vf


class ProposerSolverState(vf.State):
    proposal: str | None = None
    solution: str | None = None
    score: float = 0.0


class ProposerSolverToolset(vf.Toolset[vf.ToolsetConfig, ProposerSolverState]):
    TOOL_PREFIX = "pair"

    async def setup_task(self, task) -> None:
        self._allowed = dict(task.allowed)

    @vf.tool
    def propose(self, problem: str) -> str:
        """Record a permitted proposed task for the solver role."""
        if problem not in self._allowed:
            raise ValueError("proposal is outside this task's verified set")
        self.state.proposal = problem
        return problem

    @vf.tool
    def solve(self, answer: str) -> str:
        """Record the solver's answer to the active proposal."""
        if self.state.proposal is None:
            raise ValueError("propose before solving")
        self.state.solution = answer
        return "recorded"

    @vf.tool
    def submit(self) -> dict[str, object]:
        """Verify the proposal/solution pair objectively."""
        expected = self._allowed.get(self.state.proposal or "")
        self.state.score = float(expected is not None and self.state.solution == expected)
        return {"passed": bool(self.state.score)}


if __name__ == "__main__":
    ProposerSolverToolset.run()
