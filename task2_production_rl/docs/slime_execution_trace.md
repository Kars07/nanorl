# slime Execution Trace (Single Sample End-to-End)

This document traces a single sample through **slime** (`06ffdbe22be068b52f9ed0fc318c473f7030197e`), detailing the exact Python file, class, function name, line numbers, input data type, output data type, process boundary, and transport path for every step.

---

## 1. Step-by-Step Execution Path Matrix

| Step | Action | File & Function | Input Type | Output Type | Process Boundary | Transport Channel |
| :--- | :--- | :--- | :--- | :--- | :--- | :--- |
| **1. Prompt Fetch** | Data Loading | `slime/rollout/data_source.py` (`DataSource.get_next_batch` L25) | Dataset config | Prompt strings & version $N$ | Controller Internal | Memory |
| **2. Rollout Gen** | RadixAttention Decode| `slime/rollout/sglang_rollout.py` (`generate_rollout` L35) | Prompt strings | Tokens & sampling logprobs | Controller $\to$ SGLang Engine | SGLang HTTP / gRPC Router |
| **3. Reward Eval** | Verifier Execution | `slime/agent/sandbox.py` (`Sandbox.evaluate` L15) | Completion string | Scalar reward $r_i$ | SGLang $\to$ Sandbox | Subprocess / HTTP |
| **4. Buffer Push** | Advantage & Store | `slime_plugins/rollout_buffer/buffer.py` (`app.post('/add_data')` L14) | Trajectories & rewards | Group advantage $A_i$ | Rollout Worker $\to$ Buffer Server | HTTP REST API |
| **5. Mini-Batch** | Buffer Sample Fetch | `slime_plugins/rollout_buffer/buffer.py` (`app.get('/get_batch')` L100) | Batch request | Mini-batch dict | Buffer Server $\to$ Megatron Ranks| HTTP REST API |
| **6. Training** | 3D Loss & Backward | `slime/ray/train_actor.py` (`TrainActor.train_step` L23) | Mini-batch tensors | Gradients $\nabla_\theta \mathcal{L}$ | Megatron Distributed Ranks | Megatron Intra-Rank NCCL |
| **7. Delta Sync** | Weight Encoding | `slime/utils/disk_delta.py` (`overwrite_encode` L21) | Updated state dict | Disk delta byte stream | Megatron Ranks $\to$ SGLang Engines| Disk Filesystem / SGLang `/pull_weights` |

---

## 2. Detailed Data Type Specifications

### Step 4 Rollout Buffer Payload (`slime_plugins/rollout_buffer/buffer.py`)
```python
class RolloutDataPayload(BaseModel):
    instance_id: str
    prompt_tokens: list[int]
    completion_tokens: list[int]
    sampling_logprobs: list[float]
    reward: float
    policy_version: int
```

### Step 7 Disk Delta Weight Transfer Encoding (`slime/utils/disk_delta.py#L21-26`)
```python
def overwrite_encode(new: np.ndarray, changed_mask: np.ndarray) -> np.ndarray:
    """The 'overwrite' delta: changed-position count (u4), positions (u4 each), then new values."""
    pos = np.flatnonzero(changed_mask).astype("<u4")
    return np.concatenate([np.array([pos.size], "<u4").view(np.uint8), pos.view(np.uint8), new[changed_mask]])
```
