from __future__ import annotations

import json
from pathlib import Path
from typing import Any

DEFAULT_SUMMARY = Path("artifacts/evals/self_hosted_baseline_summary.json")


def load_traces(path: Path = DEFAULT_SUMMARY) -> list[dict[str, Any]]:
    """Load traces from a Verifiers JSONL file or the Modal summary artifact."""
    text = path.read_text(encoding="utf-8")
    if path.name.endswith("summary.json"):
        summary = json.loads(text)
        matches = [
            value
            for name, value in summary.get("artifacts", {}).items()
            if name.endswith("traces.jsonl")
        ]
        if not matches:
            raise ValueError(f"no traces.jsonl embedded in {path}")
        text = matches[-1]
    traces: list[dict[str, Any]] = []
    for line in text.splitlines():
        if not line.strip():
            continue
        run = json.loads(line)
        if isinstance(run, dict) and "traces" in run:
            traces.extend(run["traces"])
        elif isinstance(run, dict) and "task" in run and "calls" in run:
            traces.append(run)
        else:
            raise ValueError(f"unrecognized trace JSONL record in {path}")
    return traces


def selected_trace(path: Path = DEFAULT_SUMMARY, index: int = 0) -> dict[str, Any]:
    traces = load_traces(path)
    if not traces:
        raise ValueError(f"no traces in {path}")
    return traces[index]
