# Repository Architecture Overview

```text
task1_sft_grpo/
├── pyproject.toml
├── uv.lock
├── README.md
├── configs/
│   ├── sft_debug.yaml
│   ├── sft_overfit.yaml
│   ├── sft_train.yaml
│   └── grpo_debug.yaml
├── data/
│   ├── raw/
│   ├── processed/
│   └── fixtures/
├── src/
│   └── sft_lab/
│       ├── __init__.py
│       ├── config.py          # Pydantic validated configs
│       ├── seed.py            # Determinism helper & env report
│       ├── model.py           # Qwen2.5-0.5B-Instruct loader
│       ├── dataset.py         # Conversation validator & PyTorch dataset
│       ├── collator.py        # Dynamic batch collator
│       ├── masking.py         # Assistant loss mask & token inspection
│       ├── metrics.py         # Manual CE loss & per-token logits decomposition
│       ├── hooks.py           # Native PyTorch activation & grad hooks
│       ├── checkpointing.py   # Checkpoint save/reload
│       ├── generation.py      # Greedy and sampling generation
│       └── grpo/
│           ├── __init__.py
│           ├── types.py       # RolloutItem & RolloutBatch containers
│           ├── logprobs.py    # Per-token logprob gathering
│           ├── rewards.py     # Math exact-match verifier
│           ├── advantages.py # Group-relative z-score advantages
│           ├── kl.py          # Schulman KL divergence estimator
│           ├── objective.py   # Clipped surrogate GRPO loss
│           └── trainer.py     # From-scratch GRPO trainer
├── scripts/
│   ├── inspect_dataset.py
│   ├── inspect_template.py
│   ├── inspect_tokens.py
│   ├── inspect_loss_mask.py
│   ├── inspect_batch.py
│   ├── inspect_logits.py
│   ├── inspect_gradients.py
│   ├── inspect_activations.py
│   ├── overfit_test.py
│   ├── compare_checkpoints.py
│   ├── eval_generation.py
│   ├── train_sft.py
│   └── train_grpo.py
├── tests/                     # Automated pytest suite
└── docs/                      # Educational walkthroughs & playbooks
```
