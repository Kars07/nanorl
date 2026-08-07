# Task 1 — SFT Microscope + From-Scratch GRPO Laboratory

Production-quality learning laboratory built around `Qwen/Qwen2.5-0.5B-Instruct` to inspect every step of Supervised Fine-Tuning (SFT) and Group Relative Policy Optimization (GRPO).

---

## 1. Reproduction Commands

### Environment & Inspection
```bash
# Lock and install dependencies
uv lock
python -m pip install -e .

# Generate environment report
python scripts/generate_env_report.py

# Run dataset, chat template, loss mask, and token inspectors
python scripts/inspect_dataset.py --data_path data/processed/sft_data.jsonl
python scripts/inspect_template.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_tokens.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_loss_mask.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_batch.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_logits.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_gradients.py --data_path data/fixtures/valid.jsonl
python scripts/inspect_activations.py --data_path data/fixtures/valid.jsonl
```

### SFT Experiments & Evaluation
```bash
# Run tiny overfit test
python scripts/overfit_test.py

# Compare trained checkpoint vs base model
python scripts/compare_checkpoints.py --base_dir Qwen/Qwen2.5-0.5B-Instruct --trained_dir artifacts/checkpoints/sft_overfit

# Evaluate generation quality
python scripts/eval_generation.py --model_dir artifacts/checkpoints/sft_overfit

# Run full SFT training
python scripts/train_sft.py --config configs/sft_train.yaml
```

### GRPO Experiments
```bash
# Run from-scratch GRPO experiment
python scripts/train_grpo.py --config configs/grpo_debug.yaml --num_steps 3
```

### Verification & Testing
```bash
# Run full unit test suite
uv run pytest -q

# Run ruff linter & formatter checks
uv run ruff check .
uv run ruff format --check .
```

---

## 2. Learner Walkthroughs & Documentation

1. [`docs/architecture.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/docs/architecture.md): Repository structure breakdown
2. [`docs/sft_walkthrough.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/docs/sft_walkthrough.md): SFT raw example to cross-entropy & parameter update walkthrough
3. [`docs/grpo_walkthrough.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/docs/grpo_walkthrough.md): GRPO rollout to advantage & clipped surrogate loss walkthrough
4. [`docs/grpo_custom_vs_trl.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/docs/grpo_custom_vs_trl.md): Feature matrix comparing custom GRPO vs TRL GRPOTrainer
5. [`docs/debugging_playbook.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/docs/debugging_playbook.md): SFT and RL debugging playbook
6. [`bug_lab/README.md`](file:///C:/Users/eniai/OneDrive/Desktop/learnml/task1_sft_grpo/bug_lab/README.md): 20 deliberate SFT bug injection cases
