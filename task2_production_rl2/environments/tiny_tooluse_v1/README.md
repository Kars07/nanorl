# tiny-tooluse-v1

A v1 verifiers environment, scaffolded with `init`.

## Develop

1. Implement `load` and the `@reward` in `tiny_tooluse_v1/taskset.py` (see `environments/*_v1`).
2. Install + run:

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval tiny-tooluse-v1 -n 3    # evaluate a few tasks with the bash harness
```

## Layout

- `tiny_tooluse_v1/taskset.py` — the task (`@reward` scoring + behavior) and the taskset: `load` (data + prompts).
- `tiny_tooluse_v1/servers/tool.py` — a `vf.Toolset` tool server, constructed in `Task.toolsets`.

Tune knobs from the CLI: `--env.taskset.num-tasks 10`, `--model <id>`, `-n`, and `-r`.
