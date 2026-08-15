# tiny-repo-repair-v1

A native Verifiers v1 taskset with 150 deterministic repository-repair instances, split as
100 train / 25 validation / 25 test. The built-in
`null` harness sees only MCP tools; those tools execute in a secure E2B microVM. The model never
receives the E2B key or the hidden checker command.

```bash
uv sync
uv run eval tiny-repo-repair-v1 --dry-run -n 3 \
  --env.agent.harness.id null --env.agent.runtime.type subprocess
```

`TaskData` contains immutable seed files and checker metadata. `RepairState` contains mutable,
rollout-local E2B results. The weighted reward is the hidden final-state checker result.
