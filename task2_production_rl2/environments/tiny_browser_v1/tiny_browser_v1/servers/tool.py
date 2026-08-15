from __future__ import annotations

import re

import verifiers.v1 as vf
from pydantic import Field


class BrowserState(vf.State):
    visited: list[str] = Field(default_factory=list)
    answer: str | None = None
    score: float = 0.0


class TinyBrowserToolset(vf.Toolset[vf.ToolsetConfig, BrowserState]):
    TOOL_PREFIX = "browser"

    async def setup_task(self, task) -> None:
        self._pages = dict(task.pages)
        self._answer = task.answer

    @vf.tool
    def search(self, query: str) -> list[str]:
        """Search deterministic local pages by case-insensitive term."""
        terms = {term.strip("?.,!:'\"").casefold() for term in query.split() if len(term) > 2}
        return [
            name
            for name, body in self._pages.items()
            if terms.intersection(f"{name} {body}".casefold().split())
        ]

    @vf.tool
    def open_page(self, page: str) -> str:
        """Open one deterministic local page."""
        if page not in self._pages:
            matches = self.search(page)
            if len(matches) != 1:
                raise KeyError(page)
            page = matches[0]
        body = self._pages[page]
        self.state.visited.append(page)
        return body

    @vf.tool
    def submit(self, answer: str) -> dict[str, object]:
        """Submit an answer against the task's hidden reference."""
        if not self.state.visited:
            return {
                "passed": False,
                "error": "evidence required: call browser_open_page with a page returned by browser_search",
            }
        self.state.answer = answer
        normalized = {"zero": "0", "one": "1", "two": "2", "three": "3"}
        cleaned = answer.strip().casefold()
        numbers = re.findall(r"\b\d+\b", cleaned)
        actual = numbers[-1] if numbers else normalized.get(cleaned, cleaned)
        if not numbers:
            for word, number in normalized.items():
                if re.search(rf"\b{word}\b", cleaned):
                    actual = number
                    break
        expected = normalized.get(self._answer.casefold(), self._answer.casefold())
        self.state.score = float(actual == expected)
        return {"passed": bool(self.state.score)}


if __name__ == "__main__":
    TinyBrowserToolset.run()
