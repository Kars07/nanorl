from __future__ import annotations

import argparse
import json
from pathlib import Path


def inspect(path: Path) -> dict[str, object]:
    """Inspect sampled/context interleaving in an actual Prime-merged sample."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    reports: list[dict[str, object]] = []
    for index, sample in enumerate(payload["samples"]):
        mask = sample["mask"]
        spans: list[list[int]] = []
        start: int | None = None
        for position, enabled in enumerate([*mask, False]):
            if enabled and start is None:
                start = position
            elif not enabled and start is not None:
                spans.append([start, position])
                start = None
        gaps = [[left[1], right[0]] for left, right in zip(spans, spans[1:], strict=False)]
        reports.append(
            {
                "sample": index,
                "token_count": len(sample["token_ids"]),
                "sampled_spans": spans,
                "context_gaps_between_calls": gaps,
                "four_call_interleaving": payload["model_calls"] == len(spans) == 4,
                "arrays_aligned": len(sample["token_ids"])
                == len(mask)
                == len(sample["inference_logprobs"])
                == len(sample["advantages"]),
            }
        )
    return {"trace_id": payload["trace_id"], "samples": reports}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("path", nargs="?", type=Path, default=Path("artifacts/samples/repo_repair_training_sample.json"))
    print(json.dumps(inspect(parser.parse_args().path), indent=2))


if __name__ == "__main__":
    main()
