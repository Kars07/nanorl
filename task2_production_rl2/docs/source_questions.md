# Source-reading answers

1. Tasksets load through Verifiers taskset loading/registry code on the client/orchestrator.
2. The client owns lazy task construction; workers own concrete task execution.
3. Task data/config are serialized through the environment server/client boundary.
4. `Trace` is created for an episode in the worker rollout path.
5. The harness model URL is an interception server that records model calls.
6. The harness program executes in its configured runtime.
7. `mcp/launch.py` starts tool servers and supplies URLs to MCP-capable harnesses.
8. `Env` owns control flow/agents; `Taskset` owns available work.
9. Prime's orchestrator renderer/rollout pipeline turns traces into samples.
10. Sampling logprobs live on sampled token/node data and wire training records.
11. `GRPOAlgorithm.score_group` assigns group advantages.
12. Orchestrator pre/post-batch filters run around group finalization/batch assembly.
13. Transport records send IDs, masks, logprobs, advantages, and loss-weight streams.
14. Trainer logprobs are computed during the RL forward pass.
15. `trainer/rl/loss.py` computes `exp(trainer - inference)`.
16. Default is direction-masked DPPO policy gradient plus squared-log-ratio KL.
17. Component membership/weights and packed-token normalization are explicit streams.
18. Observation tokens remain context and are masked unless an algorithm routes CE credit.
19. Successive rendered steps merge only when the extension property holds.
20. The renderer/sample builder compares prior tokens with the later prefix.
21. A failed extension begins a new training sample.
22. Discontinuous later steps train in separate samples with their own masks.
23. Policy version is stamped on dispatched groups/rollouts.
24. Rollout collection overlaps trainer steps; age measures the overlap.
25. Trainer broadcasts weights and orchestrator coordinates inference update.
26. Single-node supported non-LoRA runs default to NCCL.
27. Filesystem is a fallback and persists step weight files.
28. Prime's vLLM extension implements update/load endpoints.
29. Trainer state, optimizer/scheduler/progress, orchestrator state, and exported weights.
30. Resume loads a complete checkpoint and re-establishes inference policy state.
