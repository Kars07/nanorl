# Prime-RL Execution Trace (Single GRPO Batch Step-by-Step)

This document traces a single prompt through **Prime-RL** (`ec92686fbceb9375d2155cd05c6e87652bf68441`), identifying the exact Python file, class, function name, line numbers, input data type, output data type, process boundary, and transport path for every step.

---

## 1. Step-by-Step Execution Path Matrix

| Step | Action | File & Function | Input Type | Output Type | Process Boundary | Transport Channel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Dispatch** | Prompt Selection | `prime_rl/orchestrator/dispatcher.py` (`RolloutDispatcher.dispatch_next_batch` L15) | `dict[str, Any]` (prompt) | `dict` (formatted prompt) | Orchestrator Internal | Memory |
| **2. Generate** | Token Decoding | `prime_rl/inference/vllm/server.py` (`AsyncLLMEngine.generate` L50) | `dict` (token IDs) | `Rollout` completion tokens & logprobs | Orchestrator $\to$ Inference | HTTP / ZMQ (`transport/zmq.py`) |
| **3. Reward** | Verifier Scoring | `prime_rl/orchestrator/envs.py` (`TrainEnv._sampling` L100) | `Rollout` completion string | `float` reward $r_i$ | Orchestrator Internal | Memory (`verifiers` pkg) |
| **4. Advantage**| Group Baseline | `prime_rl/orchestrator/algo/grpo.py` (`GRPOAlgorithm.compute_advantages` L45) | `list[float]` rewards | `list[float]` advantages $A_i$ | Orchestrator Internal | Memory |
| **5. Packing** | Sequence Packing | `prime_rl/trainer/rl/packer.py` (`FirstFitDecreasingPacker` L50) | Unpadded tokens & advantages | 1D Packed `TrainBatch` | Orchestrator $\to$ Trainer | ZeroMQ Queue (`transport/zmq.py`) |
| **6. Training** | Loss & Backward | `prime_rl/trainer/rl/train.py` (`compute_loss` L30 in `loss.py`) | Packed `LossInputs` (L14) | `LossOutputs` (L32) & $\nabla_\theta \mathcal{L}$ | Trainer Internal | GPU CUDA Graph |
| **7. Weight Sync**| Parameter Broadcast| `prime_rl/trainer/rl/broadcast/nccl.py` (`WeightBroadcaster` L15) | State dict tensors $\theta_{N+1}$ | Reloaded vLLM model weights | Trainer $\to$ Inference | NCCL / Shared Memory (`nixl/`) |

---

## 2. Detailed Data Type Specifications

### Step 5 Packed Batch Data Structure (`src/prime_rl/trainer/rl/packer.py`)
```python
# Packed sequence tensor format sent via ZMQ to trainer:
packed_batch = {
    "input_ids": Tensor[sum(seq_lens)],  # 1D flattened token IDs
    "cu_seqlens": Tensor[batch_size + 1],  # Cumulative sequence length bounds
    "completion_mask": Tensor[sum(seq_lens)],  # Boolean mask selecting action tokens
    "advantages": Tensor[sum(seq_lens)],  # Per-token scalar group advantages
    "old_logprobs": Tensor[sum(seq_lens)],  # Sampling log-probabilities
}
```

### Step 6 Loss Function Data Structure (`src/prime_rl/trainer/rl/loss.py#L14-35`)
```python
@dataclass
class LossInputs:
    trainer_logprobs: Float[Tensor, " seq"]
    inference_logprobs: Float[Tensor, " seq"]
    ref_logprobs: Float[Tensor, " seq"] | None
    advantages: Float[Tensor, " seq"]
    loss_mask: Bool[Tensor, " seq"]
    loss_weights: Float[Tensor, " seq"] | None = None
```
