# Pinned source map

Paths below are relative to the pinned upstream checkouts recorded in
`upstream/REVISIONS.md`.

| Concept | Source | Symbol | Responsibility | Owning process |
|---|---|---|---|---|
| TaskData | `verifiers/verifiers/v1/task.py` | `TaskData` | Immutable prompt/resources/custom fields | taskset client, serialized to worker |
| Task | `verifiers/verifiers/v1/task.py` | `Task` | setup, validation, tools, rewards, metrics | environment worker |
| Taskset | `verifiers/verifiers/v1/taskset.py` | `Taskset` | lazy task production and configuration | client/orchestrator side |
| Harness | `verifiers/verifiers/v1/harness.py` | `Harness`, `HarnessSession` | launches/continues the agent program | worker runtime |
| Agent | `verifiers/verifiers/v1/agent.py` | `Agent` | model/client/harness/runtime binding | worker |
| Env | `verifiers/verifiers/v1/env.py` | `Env` | single/multi-agent control flow | worker |
| Trace | `verifiers/verifiers/v1/trace.py` | `Trace`, `TraceNode` | live nodes, calls, state, rewards | worker; returned to client |
| ModelCall | `verifiers/verifiers/v1/clients.py` | model interception records | request/response tokens, usage, timing | interception server |
| MCP launch | `verifiers/verifiers/v1/mcp/launch.py` | `serve_tools`, `serve_in_runtime` | starts tool servers and publishes URLs | worker/runtime boundary |
| Subprocess runtime | `verifiers/verifiers/v1/runtimes/subprocess.py` | `SubprocessRuntime` | trusted local process execution | worker host; not isolation |
| Eval CLI | `verifiers/verifiers/v1/cli/eval.py` | `main` | config, pool, episodes, artifacts | eval client |
| RL launcher | `prime-rl/src/prime_rl/entrypoints/rl.py` | `rl`, `main` | validates/splits config and launches 3 processes | Modal parent process |
| Sampler | `prime-rl/src/prime_rl/orchestrator/sampler.py` | sampler helpers | inference request and trace sampling metadata | orchestrator |
| Dispatcher | `prime-rl/src/prime_rl/orchestrator/dispatcher.py` | `RolloutDispatcher` | groups, policy-version stamping, env dispatch | orchestrator |
| Algorithm | `prime-rl/src/prime_rl/orchestrator/algo/base.py` | `Algorithm` | score rollout/group and stamp loss streams | orchestrator |
| GRPO | `prime-rl/src/prime_rl/orchestrator/algo/grpo.py` | `GRPOAlgorithm` | reward-minus-group-mean advantages | orchestrator |
| Rollout/sample | `prime-rl/src/prime_rl/orchestrator/types.py` | `Rollout` | traces, samples, rewards, policy version | orchestrator → trainer |
| Interleaving | `prime-rl/src/prime_rl/orchestrator` renderer/sample code | trajectory conversion | extension checks and merged samples | orchestrator |
| Wire batch | `prime-rl/src/prime_rl/transport/types.py` | training wire records | token/loss/logprob streams | orchestrator → trainer |
| Trainer batch | `prime-rl/src/prime_rl/trainer/batch.py` | packing helpers | truncation, masks, packed microbatches | trainer |
| Default loss | `prime-rl/src/prime_rl/trainer/rl/loss.py` | `default_loss_fn` | DPPO mask, ratio PG, squared-log-ratio KL | trainer |
| Weight update | `prime-rl/src/prime_rl/trainer/rl/train.py` | `broadcast_weights` calls | exports step weights after optimizer | trainer |
| Inference update | `prime-rl/src/prime_rl/inference/vllm/server.py` | worker extension endpoints | `/update_weights`, broadcaster init | inference |
| Renderer | `renderers/renderers` | renderer classes | messages ↔ tokens and attribution | orchestrator/inference |
