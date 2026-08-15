from __future__ import annotations

import argparse
import json
import re
from pathlib import Path


ANSI = re.compile(r"\x1b\[[0-9;]*m")
STEP = re.compile(r"Step (\d+).*?Mismatch KL ([0-9.eE+-]+)")


def inspect(path: Path) -> dict[str, object]:
    """Read Prime trainer-computed mismatch KL without inventing token diagnostics."""
    clean = ANSI.sub("", path.read_text(encoding="utf-8", errors="replace"))
    rows = [{"step": int(step), "mismatch_kl": float(value)} for step, value in STEP.findall(clean)]
    if not rows:
        raise ValueError(f"no Prime trainer Mismatch KL records in {path}")
    return {
        "source": str(path),
        "metric_owner": "Prime-RL trainer",
        "steps": rows,
        "worst": max(rows, key=lambda row: float(row["mismatch_kl"])),
        "token_level_delta_available": False,
        "note": "The saved trainer log exposes aggregate Mismatch KL, not trainer token logprobs.",
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("artifacts/training/repo_repair/trainer.log"))
    print(json.dumps(inspect(parser.parse_args().path), indent=2))


if __name__ == "__main__":
    main()
