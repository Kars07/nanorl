from __future__ import annotations

import verifiers.v1 as vf


class LearningHarnessData(vf.TaskData):
    expected: str


class LearningHarnessTask(vf.Task[LearningHarnessData]):
    @vf.reward(weight=1.0)
    async def exact_answer(self, trace: vf.Trace) -> float:
        return float(trace.last_reply.strip().casefold() == self.data.expected.casefold())


class LearningHarnessConfig(vf.TasksetConfig):
    num_tasks: int = 3


class LearningHarnessTaskset(vf.Taskset[LearningHarnessTask, LearningHarnessConfig]):
    def load(self):
        cases = [
            ("Reply with exactly: amber", "amber"),
            ("Reply with exactly: cobalt", "cobalt"),
            ("Reply with exactly: jade", "jade"),
        ]
        for idx, (prompt, expected) in enumerate(cases[: self.config.num_tasks]):
            yield LearningHarnessTask(
                LearningHarnessData(
                    idx=idx,
                    prompt=prompt,
                    system_prompt="Answer the user accurately and concisely.",
                    expected=expected,
                )
            )
