from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "path",
        nargs="?",
        type=Path,
        default=Path("artifacts/samples/repo_repair_training_sample.json"),
    )
    sample = json.loads(parser.parse_args().path.read_text(encoding="utf-8"))["samples"][0]
    for name in ("rl_weights", "ce_weights", "ref_kl_weights"):
        stream = sample.get(name)
        print(f"{name}: {'implicit/default' if stream is None else f'{sum(v != 0 for v in stream)} nonzero'}")
    print(f"loss_mask_nonzero={sum(sample['mask'])} total={len(sample['mask'])}")


if __name__ == "__main__":
    main()
