from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect a Prime-RL TrainingSample artifact")
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("artifacts/samples/repo_repair_training_sample.json"),
    )
    args = parser.parse_args()
    payload = json.loads(args.path.read_text(encoding="utf-8"))
    print(
        f"trace={payload['trace_id']} calls={payload['model_calls']} "
        f"branches={payload['branches']} samples={len(payload['samples'])}"
    )
    for index, sample in enumerate(payload["samples"]):
        advantages = sample.get("advantages") or []
        print(
            f"sample={index} tokens={sample['token_count']} "
            f"trainable={sample['trainable_tokens']} context={sample['context_tokens']} "
            f"aligned={sample['aligned']} spans={sample['sampled_spans']} "
            f"advantage_min={min(advantages) if advantages else None} "
            f"advantage_max={max(advantages) if advantages else None}"
        )


if __name__ == "__main__":
    main()
