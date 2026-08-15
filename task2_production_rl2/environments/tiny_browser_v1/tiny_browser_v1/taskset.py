from __future__ import annotations

import verifiers.v1 as vf
from pydantic import Field

from tiny_browser_v1.servers.tool import BrowserState, TinyBrowserToolset


class BrowserData(vf.TaskData):
    pages: dict[str, str] = Field(default_factory=dict)
    answer: str


class BrowserTask(vf.Task[BrowserData, BrowserState]):
    @classmethod
    def toolsets(cls, config: vf.TaskConfig) -> list[vf.Toolset]:
        return [TinyBrowserToolset(vf.ToolsetConfig())]

    @vf.reward(weight=1.0)
    async def correct_answer(self, trace: vf.Trace) -> float:
        return trace.state.score

    @vf.metric
    async def pages_visited(self, trace: vf.Trace) -> float:
        return float(len(trace.state.visited))


PAGES = {
    "home": "Index: planets, missions, observatories",
    "planets": "Mars has two moons. Venus has zero moons.",
    "missions": "Aster launched in 2024. Boreal launched in 2026.",
    "observatories": "North Ridge altitude 2100m. South Array altitude 1800m.",
}
CASES = [
    ("How many moons does Mars have?", "2"),
    ("Which mission launched in 2026?", "Boreal"),
    ("Which observatory is higher?", "North Ridge"),
]


class TinyBrowserConfig(vf.TasksetConfig):
    num_tasks: int = 3


class TinyBrowserTaskset(vf.Taskset[BrowserTask, TinyBrowserConfig]):
    def load(self):
        for idx, (prompt, answer) in enumerate(CASES[: self.config.num_tasks]):
            yield BrowserTask(BrowserData(
                idx=idx,
                prompt=prompt,
                system_prompt=(
                    "This is a tool-use evaluation. First call browser_search with the key noun "
                    "from the question. Open a returned page with browser_open_page. Finally call "
                    "browser_submit with only the answer; a prose answer is not a submission."
                ),
                pages=PAGES,
                answer=answer,
            ))
