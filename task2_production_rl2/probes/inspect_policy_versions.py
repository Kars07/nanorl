from __future__ import annotations

import argparse
import re
from pathlib import Path

STEP = re.compile(r"Step (\d+).*Reward ([0-9.]+).*Max Off-Policy (\d+)")
UPDATE = re.compile(r"Updating policy in-flight to v(\d+)")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("artifacts/training/official_reverse_text/modal-command.log"),
    )
    args = parser.parse_args()
    text = args.path.read_text(encoding="utf-8", errors="replace")
    for step, reward, lag in STEP.findall(text):
        print(f"step={step} reward={reward} max_off_policy={lag}")
    versions = [int(value) for value in UPDATE.findall(text)]
    print(f"weight_updates={versions}")


if __name__ == "__main__":
    main()
