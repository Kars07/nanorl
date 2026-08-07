# slime Architecture & Process Call/Data Flow

This document presents a concrete, source-grounded breakdown of the **slime** repository (pinned at commit `06ffdbe22be068b52f9ed0fc318c473f7030197e` in `external_repos/slime`).

---

## 1. Process & Component Entry Points

slime is structured around Ray actor groups managing Megatron-LM training ranks, SGLang inference engines, and a central Rollout Buffer server:

```text
1. Ray Actor Group Controller:
   - File: slime/ray/actor_group.py (L1-200)
   - Function/Class: RayActorGroup (manages cluster lifecycle)
   - Placement Group: slime/ray/placement_group.py (L1-150)

2. Megatron Distributed Trainer Ranks:
   - File: slime/ray/train_actor.py (L1-120)
   - Function/Class: TrainActor (Megatron 3D parallel model rank)
   - Distributed Utils: slime/utils/distributed_utils.py (L1-180)
   - Delta Weight Publisher: slime/utils/disk_delta.py (L21 overwrite_encode)

3. SGLang Sampling Engine Ranks:
   - File: slime/ray/rollout.py (L1-1488)
   - File: slime/rollout/sglang_rollout.py (L1-640)
   - Function/Class: SGLangEngine (SGLang server launcher & RadixAttention manager)
   - Async Rollout: slime/rollout/fully_async_rollout.py (L1-300)

4. Data Buffer Server:
   - File: slime_plugins/rollout_buffer/buffer.py (L14-340)
   - Function/Class: FastAPI server app hosting `/add_data` and `/get_batch`
```

---

## 2. Process Architecture & Call Flow

```text
 ┌────────────────────────────────────────────────────────────────────────┐
 │                        Ray Actor Group Controller                      │
 │   `slime/ray/actor_group.py` (RayActorGroup)                           │
 │   - Manages Ray placement groups (`placement_group.py`)                │
 │   - Drives training iterations and rollout async scheduling            │
 └─────────────┬──────────────────────────────────────▲───────────────────┘
               │ (Async Prompt Dispatch)              │ (Scored Samples)
               ▼                                      │
 ┌─────────────────────────────────────────┐  ┌───────┴────────────────────────────────┐
 │           SGLang Engine Ranks           │  │       Rollout Buffer Server Plugin     │
 │ `slime/rollout/sglang_rollout.py`       │  │ `slime_plugins/rollout_buffer/buffer.py`│
 │ - Hosts SGLang engine (RadixAttention)  │  │ - Stores trajectories & rewards        │
 │ - Decodes completions & sampling logprob│  │ - Computes advantages & mini-batches   │
 └────────────────────┬────────────────────┘  └───────▲────────────────────────────────┘
                      │ (Completed Rollouts)          │
                      └───────────────────────────────┘
                                                      │ (Mini-Batches via HTTP)
                                                      ▼
                                      ┌────────────────────────────────┐
                                      │   Megatron Distributed Trainer │
                                      │ `slime/ray/train_actor.py`     │
                                      │ - 3D Parallel Model (TP/PP/DP) │
                                      │ - Loss backward & AdamW step   │
                                      │ - Encodes delta weights via    │
                                      │   `disk_delta.py` (L21)        │
                                      └────────────────────────────────┘
```

---

## 3. Parameter Passthrough & Argument Hierarchy

Argument parsing in `slime/utils/arguments.py#L35-2011` integrates three distinct framework argument parsers into a unified CLI interface:

1. **Megatron Arguments**: Model dimensions, `--tensor-model-parallel-size` (TP), `--pipeline-model-parallel-size` (PP), `--data-parallel-size` (DP), learning rate, sequence length.
2. **SGLang Arguments**: `--mem-fraction-static`, RadixAttention cache limits, server ports.
3. **slime Arguments**: `--rollout-batch-size`, `--rollout-num-workers`, buffer capacity, policy lag thresholds.
