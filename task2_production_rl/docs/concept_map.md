# Production RL Concept Map

This concept map defines 22 fundamental production reinforcement learning concepts, explaining why each exists, its state, communications, failure modes, and concrete representations in **Prime-RL** (`ec92686fbceb9375d2155cd05c6e87652bf68441`) versus **slime** (`06ffdbe22be068b52f9ed0fc318c473f7030197e`).

---

## 1. Worker
- **Definition**: An independent OS process or Ray actor executing a specific functional role in the RL loop.
- **Why it exists**: Enables parallel execution across CPU/GPU nodes.
- **Data owned**: Local process memory, socket endpoints.
- **State mutated**: Process status, local task queues.
- **Communications**: Communicates via ZeroMQ, HTTP, or Ray RPC.
- **Failure modes**: Process crash, OOM, timeout, network disconnect.
- **Prime-RL Representation**: Standalone processes launched via `prime_rl.entrypoints` (`inference.py`, `orchestrator.py`, `trainer.py`).
- **slime Representation**: Ray actors managed via `slime/ray/actor_group.py` and `slime/ray/ray_actor.py`.

---

## 2. Actor / Trainable Policy
- **Definition**: The active neural network model being updated by gradient descent.
- **Why it exists**: Represents the policy $\pi_\theta(a \mid s)$ optimized to maximize expected returns.
- **Data owned**: Trainable parameters $\theta$, gradients $\nabla_\theta \mathcal{L}$.
- **State mutated**: Parameters $\theta$ during optimizer steps.
- **Communications**: Sends updated weights to inference servers; receives packed rollout batches from orchestrator.
- **Failure modes**: Gradient explosion, NaNs, optimizer divergence, weight broadcast timeout.
- **Prime-RL Representation**: PyTorch FSDP2 model inside `prime_rl.trainer.rl.train`.
- **slime Representation**: Megatron-LM distributed training rank in `slime/ray/train_actor.py`.

---

## 3. Reference / Frozen Policy
- **Definition**: A static snapshot of the pre-trained or SFT base model $\pi_{\text{ref}}$.
- **Why it exists**: Used to compute KL divergence penalties ($\mathbb{D}_{\text{KL}}(\pi_\theta \parallel \pi_{\text{ref}})$) to prevent policy collapse.
- **Data owned**: Frozen parameters $\theta_{\text{ref}}$.
- **State mutated**: None (read-only).
- **Communications**: Evaluated during rollout or reference scoring.
- **Failure modes**: VRAM exhaustion if co-located without offloading.
- **Prime-RL Representation**: Reference scoring path in `prime_rl.orchestrator.algo`.
- **slime Representation**: Static reference weights evaluated via SGLang or OPD rollout in `slime/rollout/on_policy_distillation.py`.

---

## 4. Rollout Engine
- **Definition**: High-throughput inference server optimized for batched text generation (e.g. vLLM or SGLang).
- **Why it exists**: Standard PyTorch training forward passes are far too slow for token generation at scale.
- **Data owned**: KV cache memory pool, loaded model weights.
- **State mutated**: KV cache during generation, weights upon receiving sync broadcast.
- **Communications**: Receives prompts from orchestrator/router; returns generated completion tokens and sampling logprobs.
- **Failure modes**: KV cache OOM, batching latency spikes, weight sync deadlocks.
- **Prime-RL Representation**: `vLLM` server in `prime_rl.inference.vllm.server`.
- **slime Representation**: `SGLang` engine in `slime/rollout/sglang_rollout.py`.

---

## 5. Reward Computation
- **Definition**: Verifier function or reward model scoring generated completions.
- **Why it exists**: Provides scalar feedback signal $r_i \in \mathbb{R}$.
- **Data owned**: Rule-based verifiers or reward model weights.
- **State mutated**: None (deterministic verifier) or reward model activations.
- **Communications**: Receives `(prompt, completion)`; outputs scalar reward.
- **Failure modes**: Reward hacking, unparseable completion formats, execution timeouts in sandboxes.
- **Prime-RL Representation**: `verifiers` package integration in `prime_rl.orchestrator.envs`.
- **slime Representation**: Reward workers/verifiers in `examples/coding_agent_rl/` or `slime/agent/sandbox.py`.

---

## 6. Advantage Computation
- **Definition**: Mathematical transformation converting raw rewards $r_i$ into baseline-subtracted signals $A_i$.
- **Why it exists**: Reduces variance in policy gradient estimation.
- **Data owned**: Prompt group reward arrays.
- **State mutated**: None.
- **Communications**: Computed in orchestrator before trajectory packing.
- **Failure modes**: Division by zero when group std is 0 (mitigated by $\epsilon$).
- **Prime-RL Representation**: `prime_rl.orchestrator.algo.grpo` advantage calculation.
- **slime Representation**: Rollout buffer advantage calculation in `slime_plugins/rollout_buffer/buffer.py`.

---

## 7. Trainer
- **Definition**: Distributed training process running loss backward passes and optimizer updates.
- **Why it exists**: Computes gradients across sharded parameter states (FSDP2 or Megatron).
- **Data owned**: Distributed model shards, optimizer states, gradient buffers.
- **State mutated**: Model parameters $\theta$, optimizer momentum/variance buffers.
- **Communications**: Pulls rollout batches from transport queue; pushes weight updates to inference servers.
- **Failure modes**: Process desynchronization, NCCL communication error, CUDA OOM.
- **Prime-RL Representation**: FSDP2 PyTorch trainer in `prime_rl.entrypoints.trainer`.
- **slime Representation**: Megatron-LM training loop in `slime/ray/train_actor.py`.

---

## 8. Optimizer State
- **Definition**: Auxiliary tensors maintained by optimizers like AdamW ($m_t, v_t$).
- **Why it exists**: Enables adaptive per-parameter learning rates.
- **Data owned**: Momentum and variance vectors per parameter element.
- **State mutated**: Updated every `optimizer.step()`.
- **Communications**: Saved to disk during checkpointing; not transmitted to inference workers.
- **Failure modes**: Stale state restoration after crash recovery, VRAM overhead (2x parameter size).
- **Prime-RL Representation**: FSDP2 sharded optimizer in `prime_rl.trainer.optim`.
- **slime Representation**: Megatron-LM DistributedOptimizer state.

---

## 9. FSDP / FSDP2
- **Definition**: Fully Sharded Data Parallelism (PyTorch native).
- **Why it exists**: Shards parameters, gradients, and optimizer states across data-parallel ranks.
- **Data owned**: Local parameter shards.
- **State mutated**: All-gathers parameters for forward/backward, reduces gradients.
- **Communications**: Ring / All-Gather and Reduce-Scatter collective communications via NCCL.
- **Failure modes**: Inter-GPU interconnect bandwidth bottleneck, shape mismatch during unshard.
- **Prime-RL Representation**: Primary distributed trainer backend (`prime_rl.trainer.rl.train`).
- **slime Representation**: Alternative to Megatron (slime primarily uses Megatron-LM).

---

## 10. Megatron-style Parallel Training
- **Definition**: 3D Parallelism framework combining Tensor Parallelism (TP), Pipeline Parallelism (PP), and Data Parallelism (DP).
- **Why it exists**: Scales LLM training to thousands of GPUs for large models (>7B parameters).
- **Data owned**: Tensor-parallel parameter slices and pipeline layer blocks.
- **State mutated**: Local slice weights during backward step.
- **Communications**: High-frequency intra-node TP communications and inter-node PP communications.
- **Failure modes**: High setup complexity, pipeline bubble overhead.
- **Prime-RL Representation**: Not primary backend (Prime-RL focuses on FSDP2).
- **slime Representation**: Core distributed training engine in `slime/ray/train_actor.py`.

---

## 11. Inference Server
- **Definition**: Standalone process serving HTTP/gRPC endpoints for model completion generation.
- **Why it exists**: Encapsulates model execution with continuous batching and PagedAttention.
- **Data owned**: Loaded model weights, request queues.
- **State mutated**: Serves completion requests.
- **Communications**: Endpoint requests from orchestrator; receives weight sync calls from trainer.
- **Failure modes**: Request queue backpressure, worker disconnect.
- **Prime-RL Representation**: vLLM serving server in `prime_rl.inference.vllm.server`.
- **slime Representation**: SGLang server in `slime/rollout/sglang_rollout.py`.

---

## 12. vLLM
- **Definition**: Open-source high-throughput LLM serving engine featuring PagedAttention.
- **Why it exists**: Efficient memory management for KV cache during decoding.
- **Data owned**: Paged KV cache blocks.
- **State mutated**: KV cache allocation.
- **Communications**: HTTP / ZMQ.
- **Failure modes**: Unsupported custom model architectures, VRAM fragmentation.
- **Prime-RL Representation**: Default rollout engine backend.
- **slime Representation**: Supported as alternative rollout backend (`slime/rollout/vllm_rollout.py`).

---

## 13. SGLang
- **Definition**: Fast serving engine and programming model for LLMs featuring RadixAttention.
- **Why it exists**: Reuses KV cache across complex prompt structures and multi-turn workflows.
- **Data owned**: Radix tree KV cache.
- **State mutated**: Radix tree nodes.
- **Communications**: SGLang HTTP / ZMQ router.
- **Failure modes**: Cache invalidation bugs, routing latency.
- **Prime-RL Representation**: Supported via custom endpoints.
- **slime Representation**: Default rollout engine backend (`slime/rollout/sglang_rollout.py`).

---

## 14. Weight Synchronization
- **Definition**: Transfer mechanism updating inference engine weights with newly trained parameters.
- **Why it exists**: Ensures rollout generation reflects recent policy updates (on-policy RL).
- **Data owned**: State dict tensors / diffs.
- **State mutated**: Inference model parameters.
- **Communications**: Distributed NCCL, IPC shared memory, ZMQ, or disk filesystem.
- **Failure modes**: Weight copy race conditions, mismatched parameter shapes, silent dropped updates.
- **Prime-RL Representation**: `prime_rl.inference.vllm.worker.weight_transfer` (supports NCCL, shared memory, filesystem).
- **slime Representation**: `slime/utils/disk_delta.py` and Ray actor weight broadcast.

---

## 15. GPU Placement
- **Definition**: Allocation mapping assigning specific process ranks to physical GPU IDs.
- **Why it exists**: Prevents VRAM collisions and optimizes NVLink/PCIe interconnect usage.
- **Data owned**: `CUDA_VISIBLE_DEVICES` environment variables.
- **State mutated**: Hardware device assignment.
- **Communications**: Set during launcher process spawning.
- **Failure modes**: Device ordinal mismatch, multiple processes grabbing GPU 0.
- **Prime-RL Representation**: Configured via Slurm templates (`single_node_rl.sbatch.j2`) or env vars.
- **slime Representation**: Managed via Ray placement groups in `slime/ray/placement_group.py`.

---

## 16. Colocation
- **Definition**: Running both trainer and inference engine on the same physical GPU(s).
- **Why it exists**: Saves GPU hardware when total VRAM permits.
- **Data owned**: Shared VRAM pool.
- **State mutated**: GPU memory allocation.
- **Communications**: Inter-process communication on same host.
- **Failure modes**: Mutual VRAM starvation during peak forward/backward pass.
- **Prime-RL Representation**: Co-located mode support in single-node launch configs.
- **slime Representation**: Co-located mode supported in `slime/ray/actor_group.py`.

---

## 17. Data Buffer / Rollout Queue
- **Definition**: Queue or buffer storing generated rollout trajectories awaiting training.
- **Why it exists**: Decouples asynchronous generation from trainer consumption steps.
- **Data owned**: Lists of trajectory records.
- **State mutated**: Enqueues new rollouts; dequeues training batches.
- **Communications**: Receives items from orchestrator; served to trainer.
- **Failure modes**: Buffer overflow, memory leakage from unconsumed samples.
- **Prime-RL Representation**: Transport queues in `prime_rl.transport.zmq` / orchestrator dispatcher.
- **slime Representation**: `slime_plugins/rollout_buffer/buffer.py`.

---

## 18. Async Training
- **Definition**: Paradigm where rollouts are generated concurrently while trainer executes optimizer steps.
- **Why it exists**: Eliminates GPU idle time caused by waiting for generation steps.
- **Data owned**: Async queue buffers and policy version counters.
- **State mutated**: Asynchronous parameter updates.
- **Communications**: Non-blocking RPC / ZMQ queue pushes.
- **Failure modes**: Excessive policy lag leading to severe off-policy gradient divergence.
- **Prime-RL Representation**: Fully async orchestrator modes (`prime_rl.orchestrator.orchestrator`).
- **slime Representation**: `slime/rollout/fully_async_rollout.py`.

---

## 19. Policy Version
- **Definition**: Monotonically increasing integer $N$ incremented after each trainer optimizer step.
- **Why it exists**: Tracks exact lineage of model parameters used to generate rollouts.
- **Data owned**: Integer counter $N$.
- **State mutated**: Incremented on `optimizer.step()`.
- **Communications**: Tagged on rollout trajectories and weight sync broadcasts.
- **Failure modes**: Counter desynchronization across processes.
- **Prime-RL Representation**: `policy_version` metadata in orchestrator and batch items.
- **slime Representation**: Policy version tagging in `slime/ray/rollout.py`.

---

## 20. Rollout Staleness / Policy Lag
- **Definition**: The difference $\text{lag} = N_{\text{current\_policy}} - N_{\text{rollout\_policy}}$.
- **Why it exists**: Meaures how far off-policy a rollout sample is when consumed by trainer.
- **Data owned**: Computed integer lag metric.
- **State mutated**: None.
- **Communications**: Logged in trainer metrics.
- **Failure modes**: Extreme lag ($\text{lag} \gg 5$) causing loss instability or clipping 100% of tokens.
- **Prime-RL Representation**: Tracked in `prime_rl.orchestrator.metrics`.
- **slime Representation**: Logged in `slime/rollout/fully_async_rollout.py`.

---

## 21. Checkpoint
- **Definition**: Persistent disk artifact saving model weights, optimizer states, and trainer metadata.
- **Why it exists**: Enables crash recovery and model evaluation after training.
- **Data owned**: Disk files (`model.safetensors`, `optimizer.pt`, `config.json`).
- **State mutated**: Written to disk periodically.
- **Communications**: Disk I/O.
- **Failure modes**: Disk full error, partial/corrupted write.
- **Prime-RL Representation**: `prime_rl.trainer.ckpt` and `prime_rl.orchestrator.ckpt`.
- **slime Representation**: Megatron checkpoint saver in `slime/utils/distributed_utils.py`.

---

## 22. Orchestrator
- **Definition**: Central CPU process coordinating data movement between environment, inference, verifier, and trainer.
- **Why it exists**: Acts as the brain of the RL system, managing workflow execution and trajectory packing.
- **Data owned**: Active task queues, environment instances, reward metrics.
- **State mutated**: Workflow state.
- **Communications**: ZeroMQ / HTTP calls to inference, verifiers, and trainer.
- **Failure modes**: CPU bottlenecking, socket connection drop.
- **Prime-RL Representation**: `prime_rl.orchestrator.orchestrator` process.
- **slime Representation**: Ray controller / driver in `slime/ray/actor_group.py`.
