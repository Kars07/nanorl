from __future__ import annotations

import json
import subprocess


def main() -> None:
    query = "index,name,memory.total,memory.used,utilization.gpu"
    result = subprocess.run(
        ["nvidia-smi", f"--query-gpu={query}", "--format=csv,noheader,nounits"],
        text=True,
        capture_output=True,
    )
    rows = []
    for line in result.stdout.splitlines():
        index, name, total, used, utilization = [part.strip() for part in line.split(",")]
        rows.append(
            {"index": int(index), "name": name, "memory_mb": int(total),
             "used_mb": int(used), "utilization_percent": int(utilization)}
        )
    print(json.dumps(rows, indent=2))
    raise SystemExit(result.returncode)


if __name__ == "__main__":
    main()
