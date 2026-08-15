from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from tiny_terminal_v1.servers.tool import TerminalState, TinyTerminalToolset


class TerminalData(vf.TaskData):
    initial_files: dict[str, str] = Field(default_factory=dict)
    expected_files: dict[str, str] = Field(default_factory=dict)


class TerminalTask(vf.Task[TerminalData, TerminalState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [TinyTerminalToolset(vf.ToolsetConfig())]

    @vf.reward(weight=1.0)
    async def final_state(self, trace: vf.Trace) -> float:
        return trace.state.score


CASES = [
    ({"status.txt": "broken"}, {"status.txt": "healthy"}, "Change status.txt to healthy."),
    ({"app.conf": "PORT=80"}, {"app.conf": "PORT=8080"}, "Set the configured port to 8080."),
    ({}, {"ready.flag": "yes"}, "Create ready.flag containing yes."),
    ({"old.txt": "remove", "keep.txt": "ok"}, {"keep.txt": "ok"}, "Delete old.txt without changing keep.txt."),
    ({"a": "1"}, {"a": "1", "b": "2"}, "Create b containing 2 and preserve a."),
]


class TinyTerminalConfig(vf.TasksetConfig):
    num_tasks: int = 5


class TinyTerminalTaskset(vf.Taskset[TerminalTask, TinyTerminalConfig]):
    def load(self):
        for idx, (initial, expected, prompt) in enumerate(CASES[: self.config.num_tasks]):
            yield TerminalTask(
                TerminalData(
                    idx=idx,
                    prompt=prompt,
                    system_prompt=(
                        "This is a tool-use evaluation. Inspect files if needed, perform the requested "
                        "mutation with terminal_write or terminal_remove, then call terminal_submit. "
                        "A prose answer is not a submission."
                    ),
                    initial_files=initial,
                    expected_files=expected,
                )
            )
