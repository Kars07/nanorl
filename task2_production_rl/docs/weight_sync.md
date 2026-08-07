# Weight Synchronization in Production RL

This document explains the mechanisms, staleness risks, and verification protocols for synchronizing weights from training ranks to inference engines.

---

## 1. Why Sampler Weights Become Stale

In decoupled RL architectures (where training ranks and inference engines run in separate OS processes or GPU clusters):

- The **Trainer** updates model parameters $\theta_{N} \to \theta_{N+1}$ during optimizer steps.
- The **Sampler** continues generating rollouts using its locally loaded state dict $\theta_{N_{\text{sampler}}}$.
- If the weight transfer is non-blocking or asynchronous, the sampler naturally generates rollouts under an older policy version ($N_{\text{sampler}} < N_{\text{trainer}}$).

---

## 2. Weight Transfer Channels

| Channel | Implementation | Bandwidth / Latency | Use Case |
| :--- | :--- | :--- | :--- |
| **Direct NCCL Broadcast** | GPU-to-GPU peer-to-peer over NVLink/InfiniBand | High speed (~100+ GB/s) | Co-located or tightly-coupled nodes |
| **Shared CUDA Memory (IPC)** | NiXL / IPC memory handles | Ultra-fast intra-node (~600+ GB/s) | Intra-node co-located GPU ranks |
| **Disk Filesystem (Safetensors)**| Periodic checkpoint save & reload | Slower (~1-5 GB/s) | Disjoint multi-node clusters |

---

## 3. Automated Weight Verification Protocol

To guarantee that an inference sampler has actually applied a weight update, use weight fingerprinting:

```python
from mismatch_probe.weight_fingerprint import compute_weight_fingerprint, assert_weight_identity

fp_trainer = compute_weight_fingerprint(trainer_model)
fp_sampler = compute_weight_fingerprint(sampler_model)

# Automated assertion:
assert_weight_identity(fp_trainer, fp_sampler)
```

---

## 4. Silent Failure Modes

1. **Dropped Notification**: The trainer publishes new weights, but the sampler HTTP/ZMQ listener drops the trigger packet, leaving the sampler running on stale weights indefinitely.
2. **Partial State Dict Copy**: Shape mismatch or un-sharded parameter tensor slice omissions during FSDP/Megatron gather operations cause sub-layer parameters to remain un-updated.
3. **In-place Gradient Overwrites**: Modifying trainer weights during in-flight inference forward passes causing CUDA memory corruption or NaN activations.
