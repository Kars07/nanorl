from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from tiny_tooluse_v1.servers.tool import TinyTooluseToolset, TooluseState


class TooluseData(vf.TaskData):
    records: dict[str, dict[str, str]] = Field(default_factory=dict)
    expected: str


class TooluseTask(vf.Task[TooluseData, TooluseState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [TinyTooluseToolset(vf.ToolsetConfig())]

    @vf.reward(weight=1.0)
    async def answer(self, trace: vf.Trace) -> float:
        return trace.state.score


RECORDS = {
    "ISSUE-1": {"owner": "Ada", "status": "open"},
    "ISSUE-2": {"owner": "Lin", "status": "closed"},
    "ISSUE-3": {"owner": "Ada", "status": "blocked"},
}
CASES = [("Who owns ISSUE-2?", "Lin"), ("What is ISSUE-3's status?", "blocked")]


class TinyTooluseConfig(vf.TasksetConfig):
    num_tasks: int = 2


class TinyTooluseTaskset(vf.Taskset[TooluseTask, TinyTooluseConfig]):
    def load(self):
        for idx, (prompt, expected) in enumerate(CASES[: self.config.num_tasks]):
            yield TooluseTask(TooluseData(
                idx=idx,
                prompt=f"{prompt} Query the exact ID named in this question, then submit the requested field.",
                system_prompt=(
                    "This is a tool-use evaluation. Call issues_query with the exact issue ID, inspect "
                    "its record, then call issues_submit with only the requested value. A prose answer "
                    "is not a submission."
                ),
                records=RECORDS,
                expected=expected,
            ))
