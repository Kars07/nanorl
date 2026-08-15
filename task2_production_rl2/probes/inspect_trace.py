from __future__ import annotations

import argparse
import json
from pathlib import Path

from trace_io import DEFAULT_SUMMARY, selected_trace


def main() -> None:
    parser = argparse.ArgumentParser(description="Print a compact Verifiers v1 trace timeline")
    parser.add_argument("path", nargs="?", type=Path, default=DEFAULT_SUMMARY)
    parser.add_argument("--index", type=int, default=0)
    args = parser.parse_args()
    trace = selected_trace(args.path, args.index)
    task = trace["task"]["data"]
    print(f"trace={trace['id']} task={task['idx']} category={task.get('category')}")
    print(f"prompt: {task['prompt']}")
    for idx, node in enumerate(trace.get("nodes", [])):
        message = node["message"]
        content = str(message.get("content", "")).replace("\n", " ")[:180]
        tools = [call.get("name") for call in (message.get("tool_calls") or [])]
        print(
            f"node={idx} parent={node.get('parent')} role={message['role']} "
            f"sampled={node.get('sampled')} tools={tools} content={content!r}"
        )
    print("rewards=" + json.dumps(trace.get("rewards", {}), sort_keys=True))
    print("metrics=" + json.dumps(trace.get("metrics", {}), sort_keys=True))
    print(f"ok={trace.get('ok')} stop={trace.get('stop_condition')} errors={len(trace.get('errors', []))}")


if __name__ == "__main__":
    main()
