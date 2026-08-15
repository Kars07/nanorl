from __future__ import annotations

import asyncio

import verifiers.v1 as vf
from pydantic import Field

from proposer_solver_v1.servers.tool import ProposerSolverState, ProposerSolverToolset


class ProposerSolverData(vf.TaskData):
    allowed: dict[str, str] = Field(default_factory=dict)


class ProposerSolverTask(vf.Task[ProposerSolverData, ProposerSolverState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [ProposerSolverToolset(vf.ToolsetConfig())]

    @vf.reward(weight=1.0)
    async def verified_pair(self, trace: vf.Trace) -> float:
        return trace.state.score


class ProposerSolverConfig(vf.TasksetConfig):
    num_tasks: int = 2


class ProposerSolverTaskset(vf.Taskset[ProposerSolverTask, ProposerSolverConfig]):
    def load(self):
        cases = [
            {"reverse abc": "cba", "double 7": "14"},
            {"uppercase codex": "CODEX", "length modal": "5"},
        ]
        for idx, allowed in enumerate(cases[: self.config.num_tasks]):
            yield ProposerSolverTask(ProposerSolverData(
                idx=idx,
                prompt=f"Choose exactly one problem from this verified mapping: {allowed!r}. Propose its exact key, solve it with the mapped value, then submit.",
                system_prompt=(
                    "Use pair_propose, pair_solve, and pair_submit. Tool arguments must "
                    "copy an exact key/value pair from the verified mapping in the prompt."
                ),
                allowed=allowed,
            ))


def _seat() -> vf.AgentConfig:
    # Harness remains an unresolved CLI concern here; constructing a plugin
    # config at import time would eagerly import platform-specific harnesses.
    return vf.AgentConfig(max_turns=4)


class ProposerSolverEnvConfig(vf.EnvConfig):
    proposer: vf.AgentConfig = _seat()
    solver: vf.AgentConfig = _seat()
    n: int = Field(2, ge=1)
    train_proposer: bool = True
    train_solver: bool = True
    max_concurrent_agents: int | None = 2


class ProposerSolverEnv(vf.Env[ProposerSolverEnvConfig]):
    """One proposer trace fans out to N independent, same-role solver traces."""

    async def setup(self, agents: vf.Agents) -> None:
        agents.proposer.trainable = self.config.train_proposer
        agents.solver.trainable = self.config.train_solver

    async def run(self, task: ProposerSolverTask, agents: vf.Agents) -> None:
        proposal = await agents.proposer.run(task)
        solver_data = task.data.model_copy(
            update={
                "prompt": (
                    "A proposer produced the following candidate:\n"
                    f"{proposal.last_reply}\n\n"
                    f"Independently choose one exact pair from {task.data.allowed!r}; "
                    "call propose with its key, solve with its value, and submit."
                )
            }
        )
        solver_task = ProposerSolverTask(solver_data, task.config)
        await asyncio.gather(*(agents.solver.run(solver_task) for _ in range(self.config.n)))

    async def finalize(self, task: vf.Task, episode: vf.Episode) -> None:
        solver_traces = [
            trace for trace in episode.traces if trace.agent.name == "solver"
        ]
        solver_passes = sum(trace.reward >= 1.0 for trace in solver_traces)
        for trace in episode.traces:
            trace.record_metric("episode_agents", float(len(episode.traces)))
            trace.record_metric("solver_passes", float(solver_passes))
