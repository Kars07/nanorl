"""List or terminate only E2B sandboxes owned by this project."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from e2b import Sandbox

OWNER = "task2-production-rl2"


def api_key() -> str:
    value = os.environ.get("E2B_API_KEY")
    if value:
        return value
    env_path = Path(__file__).resolve().parents[2] / ".env"
    for line in env_path.read_text(encoding="utf-8").splitlines():
        name, separator, candidate = line.partition("=")
        if separator and name.strip() == "e2b_api":
            return candidate.strip().strip("'\"")
    raise RuntimeError("E2B_API_KEY or workspace .env e2b_api is required")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--execute", action="store_true", help="kill matching sandboxes")
    args = parser.parse_args()
    key = api_key()
    paginator = Sandbox.list(api_key=key)
    matches = []
    while True:
        matches.extend(
            item
            for item in paginator.next_items()
            if (item.metadata or {}).get("owner") == OWNER
        )
        if not paginator.has_next:
            break
    print(f"owner={OWNER} matching={len(matches)} execute={args.execute}")
    if args.execute:
        killed = sum(bool(Sandbox._cls_kill(item.sandbox_id, api_key=key)) for item in matches)
        print(f"killed={killed}")


if __name__ == "__main__":
    main()
