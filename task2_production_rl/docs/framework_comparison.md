# Framework Comparison: Prime-RL vs. slime

This document presents a side-by-side architectural comparison between **Prime-RL** (`ec92686fbceb9375d2155cd05c6e87652bf68441`) and **slime** (`06ffdbe22be068b52f9ed0fc318c473f7030197e`).

---

## Comparative Matrix

| Concept / Aspect | Prime-RL | slime |
| :--- | :--- | :--- |
| **Distributed Trainer Engine** | PyTorch Native **FSDP2** (`fully_shard`) | **Megatron-LM** (TP / PP / DP 3D Parallelism) |
| **Inference Rollout Engine** | **vLLM** (`AsyncLLMEngine`) | **SGLang** (RadixAttention Engine) |
| **Orchestration Architecture** | CPU Orchestrator process + ZMQ Transport | **Ray** Actor Group (`actor_group.py`) |
| **Environment / Verifier Integration**| `verifiers` package integration | `slime.agent` sandboxes & verifiers |
| **Advantage Path** | `prime_rl.orchestrator.algo.grpo` | `slime_plugins.rollout_buffer.buffer` |
| **Data Buffer Path** | ZMQ Transport Queues + Sequence Packer | `RolloutBuffer` plugin (`buffer.py`) |
| **Weight Synchronization** | NCCL / Shared Memory / NiXL / Disk | Disk delta state dicts (`disk_delta.py`) / Ray RPC |
| **Async Semantics** | Orchestrator async dispatcher | `fully_async_rollout.py` |
| **GPU Placement** | Slurm / Environment variables (`CUDA_VISIBLE_DEVICES`) | Ray placement groups (`placement_group.py`) |
| **Target Scale** | Small-to-Medium models (< 14B) on FSDP2 | Large-to-Giant models (7B - 70B+) on Megatron 3D |
| **Primary Use Case** | Fast experimentation, custom verifiers, FSDP2 scale | Production multi-node Megatron training with SGLang |

---

## Key Architectural Differences

1. **Trainer Engine**: Prime-RL relies on PyTorch FSDP2 for data-parallel model sharded training, making it lightweight and easy to integrate with HuggingFace models. slime relies on Megatron-LM, allowing 3D parallelism for massive scale models.
2. **Rollout Engine**: Prime-RL uses vLLM for high-throughput decoding; slime uses SGLang for RadixAttention multi-turn prompt prefix sharing.
3. **Orchestrator**: Prime-RL uses an explicit Python CPU process managing ZMQ sockets; slime relies on Ray actors and placement groups to manage workers dynamically.
