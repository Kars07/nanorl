# Verifiers v1 architecture from the actual repo-repair trace

`TinyRepoRepairTaskset.load()` constructed task 0 on the eval client. The task data was
serialized to a worker, where `TinyRepoRepairTask` configured `E2BRepoToolset`. Verifiers
started that toolset as an MCP server in its subprocess runtime. `setup_task` created one
secure E2B microVM, seeded `solution.py` and the public smoke test, and kept the hidden
checker in server memory.

The built-in `null` harness connected to the interception endpoint and MCP URL. Each model
request passed through interception to the localhost Prime inference router. Responses and
tool results became trace nodes; four requests became four `ModelCall` records. On submit,
the MCP server ran the hidden final-state check inside E2B. Reward was read from rollout
state, not from the model's prose.

```text
Taskset(client) → Task(worker) → Agent → null Harness
                                      ↘ interception → localhost vLLM
                                       ↘ MCP server → E2B microVM
                         Trace ← nodes/calls/tool results
                         Reward ← hidden final-state check
```

The baseline artifact proves the separation: task setup took 16 seconds, agent work took
16 seconds, four model calls used 623 prompt and about 2.3K completion tokens, and reward
was 0 because the model changed the wrong file. `ok=true` means the episode infrastructure
completed, not that the task passed.
