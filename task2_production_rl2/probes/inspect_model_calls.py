from __future__ import annotations

import argparse
from pathlib import Path

from trace_io import DEFAULT_SUMMARY, selected_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Inspect every ModelCall in a Verifiers trace")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SUMMARY)
    args = parser.parse_args()
    trace = selected_trace(args.path)
    nodes = trace.get("nodes", [])
    for idx, call in enumerate(trace.get("calls", [])):
        node = nodes[call["node"]]
        usage = call.get("usage", {})
        elapsed = call.get("time", {}).get("end", 0) - call.get("time", {}).get("start", 0)
        print(
            f"call={idx} node={call['node']} model={call['model']} endpoint={call['endpoint']} "
            f"finish={call.get('finish_reason')} prompt_tokens={usage.get('prompt_tokens')} "
            f"completion_tokens={usage.get('completion_tokens')} latency={elapsed:.3f}s"
        )
        print(f"  sampling={call.get('sampling', {})}")
        print(f"  output={str(node['message'].get('content', ''))[:500]!r}")


if __name__ == "__main__":
    main()
