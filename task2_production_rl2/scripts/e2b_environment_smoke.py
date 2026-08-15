import asyncio
import json
import os

from tiny_repo_repair_v1.servers.tool import E2BRepoToolset
from tiny_repo_repair_v1.taskset import TinyRepoRepairConfig, TinyRepoRepairTaskset


async def main() -> None:
    task = next(iter(TinyRepoRepairTaskset(TinyRepoRepairConfig(num_tasks=1))))
    toolset = E2BRepoToolset(task.config.e2b)
    await toolset.setup_task(task.data)
    try:
        before = toolset.read_file("solution.py")
        toolset.write_file("solution.py", "def target(a, b):\n    return a + b\n")
        submission = toolset.submit()
        state = toolset.state.model_dump(exclude={"artifacts"})
        artifacts = {
            path: data.decode() if data is not None else None
            for path, data in toolset.state.artifacts.items()
        }
        print(
            json.dumps(
                {
                    "sandbox_id": toolset._require_sandbox().sandbox_id,
                    "before": before,
                    "submission": submission,
                    "state": state,
                    "artifacts": artifacts,
                }
            )
        )
    finally:
        await toolset._exit_stack.aclose()


if __name__ == "__main__":
    if not os.environ.get("E2B_API_KEY"):
        raise RuntimeError("E2B_API_KEY is required")
    asyncio.run(main())

