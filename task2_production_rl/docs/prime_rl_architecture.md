# Prime-RL Architecture & Process Call/Data Flow

This document presents a concrete, source-grounded breakdown of the **Prime-RL** repository (pinned at commit `ec92686fbceb9375d2155cd05c6e87652bf68441` in `external_repos/prime-rl`).

---

## 1. Process Entry Points & File Mapping

Prime-RL is organized as three independent, asynchronous processes launched via `src/prime_rl/entrypoints/`:

```text
1. Inference Server Process:
   - Entry Point Script: src/prime_rl/entrypoints/inference.py (L1-20)
   - Core Server Implementation: src/prime_rl/inference/vllm/server.py (L1-245)
   - Worker Weight Transfer: src/prime_rl/inference/vllm/worker/weight_transfer.py (L13-25)

2. Orchestrator Process:
   - Entry Point Script: src/prime_rl/entrypoints/orchestrator.py (L1-16)
   - Core Implementation: src/prime_rl/orchestrator/orchestrator.py (L30-1020)
   - Rollout Dispatcher: src/prime_rl/orchestrator/dispatcher.py (L15-713)
   - Environments & Verifiers: src/prime_rl/orchestrator/envs.py (L1-252)
   - GRPO Algorithm: src/prime_rl/orchestrator/algo/grpo.py (L1-200)

3. Trainer Process:
   - Entry Point Script: src/prime_rl/entrypoints/trainer.py (L1-26)
   - Core Trainer Implementation: src/prime_rl/trainer/rl/train.py (L1-827)
   - Loss Function & Data Structures: src/prime_rl/trainer/rl/loss.py (L14-35)
   - Trajectory Sequence Packer: src/prime_rl/trainer/rl/packer.py (L26-358)
   - Weight Broadcast: src/prime_rl/trainer/rl/broadcast/ (nccl.py, nixl/, filesystem.py)
```

---

## 2. Process Architecture & Detailed Call Flow

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                      Orchestrator Process                              │
 │  `src/prime_rl/orchestrator/orchestrator.py` (main() L980)             │
 │   - Drives environments via `envs.py` (TrainEnv L100)                  │
 │   - Dispatches prompts via `dispatcher.py` (RolloutDispatcher L15)     │
 │   - Scores rewards via `verifiers` in `envs.py`                        │
 │   - Computes Group Relative Advantages in `algo/grpo.py`               │
 │   - Packs trajectories via `trainer/rl/packer.py` (BasePacker L26)     │
 └─────────────┬──────────────────────────────────────▲───────────────────┘
               │ (ZMQ Transport)                      │ (ZMQ Transport / Weights)
               │ `src/prime_rl/transport/zmq.py`       │
               ▼                                      │
 ┌─────────────────────────────────────────┐  ┌───────┴────────────────────────────────┐
 │        Inference Server Process         │  │        Trainer Process Process         │
 │ `src/prime_rl/inference/vllm/server.py` │  │ `src/prime_rl/trainer/rl/train.py`   │
 │ - Spawns vLLM `AsyncLLMEngine`          │  │ - FSDP2 PyTorch model initialization  │
 │ - Hosts live policy & decodes tokens    │  │ - Loss in `loss.py` (`LossInputs` L14)│
 │ - Receives weight updates via           │◀─┼─ Broadcasts state dict via NCCL/NiXL   │
 │   `weight_transfer.py` (L13)            │  │   in `broadcast/nccl.py`               │
 └─────────────────────────────────────────┘  └────────────────────────────────────────┘
```

---

## 3. Detailed Component Mechanics

### A. Inference Server (`src/prime_rl/inference/vllm/server.py`)
- **CLI / Entry point**: Executed via `python -m prime_rl.entrypoints.inference --config configs/inference.yaml`.
- **vLLM Engine**: Spawns `vLLM` OpenAPI serving application.
- **Weight Transfer Reload**: `load_weights_checkpoint_layerwise(model, state_iter, model_config, vllm_config)` in `src/prime_rl/inference/vllm/worker/weight_transfer.py#L13-25` reloads parameters layerwise into GPU memory without restarting the vLLM engine process.

### B. Orchestrator (`src/prime_rl/orchestrator/orchestrator.py`)
- **CLI / Entry point**: Executed via `python -m prime_rl.entrypoints.orchestrator --config configs/orchestrator.yaml`.
- **Advantage Computation**: `prime_rl.orchestrator.algo.grpo` computes group relative z-score advantages across completions:
  $$A_{g, i} = \frac{r_{g, i} - \mu(r_g)}{\sigma(r_g) + \epsilon}$$
- **Sequence Packing**: `prime_rl.trainer.rl.packer.FirstFitDecreasingPacker` packs unpadded variable-length completion sequences into single 1D tensors to achieve 100% compute efficiency on GPUs without zero-padding waste.

### C. Trainer (`src/prime_rl/trainer/rl/train.py`)
- **CLI / Entry point**: Executed via `python -m prime_rl.entrypoints.trainer --config configs/trainer.yaml`.
- **Loss Computation Data Structure**: Defined in `src/prime_rl/trainer/rl/loss.py#L14-35`:
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
- **Weight Broadcast**: `setup_weight_broadcast()` in `src/prime_rl/trainer/rl/broadcast/__init__.py` selects either NCCL (`nccl.py`), NiXL CUDA memory handles (`nixl/`), or disk filesystem (`filesystem.py`).
