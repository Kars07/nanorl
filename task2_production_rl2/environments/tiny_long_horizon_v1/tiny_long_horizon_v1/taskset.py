from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from tiny_long_horizon_v1.servers.tool import LongState, TinyLongHorizonToolset


class LongData(vf.TaskData):
    steps: list[str] = Field(default_factory=list)


class LongTask(vf.Task[LongData, LongState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [TinyLongHorizonToolset(vf.ToolsetConfig())]

    @vf.reward(weight=1.0)
    async def ordered_completion(self, trace: vf.Trace) -> float:
        return trace.state.score


class TinyLongHorizonConfig(vf.TasksetConfig):
    num_tasks: int = 2


class TinyLongHorizonTaskset(vf.Taskset[LongTask, TinyLongHorizonConfig]):
    def load(self):
        cases = [
            ["inspect", "plan", "edit", "test", "submit"],
            ["search", "compare", "verify", "report"],
        ]
        for idx, steps in enumerate(cases[: self.config.num_tasks]):
            yield LongTask(LongData(
                idx=idx,
                prompt=f"Complete these workflow steps in order: {steps!r}. Then submit.",
                system_prompt=(
                    "This is a tool-use evaluation. Repeatedly call workflow_next_step, then immediately "
                    "call workflow_complete with the returned step. Continue until next_step returns null, "
                    "then call workflow_submit. A prose answer is not a submission."
                ),
                steps=steps,
            ))
