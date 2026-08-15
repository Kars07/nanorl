# learning-harness

A v1 verifiers environment, scaffolded with `init`.

## Develop

1. Implement `load` and the `@reward` in `learning_harness/taskset.py` (see `environments/*_v1`).
2. Install + run:

```bash
uv pip install -e .        # install this package (or register it in your project)
uv run eval learning-harness -n 3    # evaluate a few tasks with the bash harness
```

## Layout

- `learning_harness/taskset.py` — the task (`@reward` scoring + behavior) and the taskset: `load` (data + prompts).
- `learning_harness/harness.py` — a custom harness, selectable with `--env.agent.harness.id learning-harness`.

Tune knobs from the CLI: `--env.taskset.num-tasks 10`, `--model <id>`, `-n`, and `-r`.
