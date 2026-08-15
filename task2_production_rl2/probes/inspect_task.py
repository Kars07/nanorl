from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_io import DEFAULT_SUMMARY, selected_trace


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    data = selected_trace(args.path)["task"]["data"]
    safe = {key: value for key, value in data.items() if key != "check_command"}
    safe["hidden_checker_present"] = bool(data.get("check_command"))
    print(json.dumps(safe, indent=2))


if __name__ == "__main__":
    main()
