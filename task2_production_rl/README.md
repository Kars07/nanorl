# Task 2 — Production RL Systems: Prime-RL + Verifiers + Slime + Training/Inference Boundary

Production LLM reinforcement learning laboratory focusing on `Prime-RL`, `slime`, Ray worker orchestration, weight synchronization, policy staleness, and training/inference numerical mismatch probing.

## Quick Start

```bash
uv lock
python -m pip install -e .
uv run pytest -q
```
