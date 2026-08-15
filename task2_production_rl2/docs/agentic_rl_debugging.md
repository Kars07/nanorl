# Agentic RL debugging playbook

| Layer | Symptom | Probe | Likely cause | Fix / regression |
|---|---|---|---|---|
| Environment | hidden check always fails | inspect submitted artifacts | impossible or wrong checker | direct E2B smoke plus known-good edit |
| Environment | correct browser fact scores zero | inspect submit argument and visited-page metric | overly exact answer normalization or no evidence navigation | normalize numeric surface forms and require `visited` before reward |
| Environment | path duplicated | trace tool error | absolute path treated relative | path-boundary unit tests |
| Sandbox | MCP setup fails | trace `ToolsetError` tail | E2B transport/dependency mismatch | signature probe and compatibility test |
| Sandbox | network unexpectedly works | execute a denied request | wrong E2B policy | create with `allow_internet_access=false`, connectivity test |
| Harness | no tools | inspect trace tool schema | harness lacks MCP support | compile-time capability assertion |
| Harness | host command exposure | inspect agent tools/runtime | unsafe harness/runtime pair | null harness plus E2B-only MCP |
| Trace | no calls | inspect setup/errors | failure before agent | `inspect_trace.py`, MCP logs |
| Renderer | trajectory splits | inspect rendered prefixes | compaction/template/handoff | treat discontinuity as new sample |
| Reward | all equal | group rewards/advantages | task too hard/easy or group 1 | improve curriculum; group size >1 |
| Inference | no health | inference log and `/health` | missing router/wheel/model load | pinned router and health gate |
| Trainer | no update | step/grad/log files | zero advantages or no trainable tokens | assert streams and group diversity |
| Off-policy | extreme ratio | exported token diagnostics | stale policy/weight mismatch | check policy version, weights, renderer, then numerics |
| Checkpoint | cannot resume | checkpoint inventory | incomplete or weights-only path | require complete checkpoint and STABLE export |
| Config | hierarchical GRPO says package missing | read validator import and probe module | versioned package does not export legacy `proposer_solver` module | re-export the identical EnvConfig class, then rerun Prime dry-run |

Observed failures are retained as evidence: NVIDIA base pull, Windows console encoding,
lock/extras mismatch, missing router, missing flash-attn, E2B transport mismatch, and unsafe
absolute-path normalization. Each was fixed at its owning layer.
