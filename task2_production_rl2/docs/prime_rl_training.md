# Prime-RL training runbook

The control is the upstream `examples/basic/reverse-text/rl.toml` without changed training
values. On Modal `A10:2` it completed 20 steps in about five minutes of orchestrator loop
time after cold startup. Each step trained 128/128 rollouts, errors stayed at 0%, late
reward was roughly 0.74–0.82, and final checkpoint/weights were written at step 20.

The agentic smoke uses `configs/rl/repo_repair_smoke.toml`. It is intentionally small:
group 4, batch 8, two steps, short tasks, eight turns, and explicit held-out source. It is
not a benchmark-quality model run. A quality experiment must expand train/validation/test,
run baseline/SFT/GEPA comparisons, schedule held-out evaluation, and reload the final
checkpoint for evaluation.

Resume must point Prime at a complete checkpoint directory, not merely HF-format weights.
The `STABLE` marker identifies a completed weight export. Never overwrite a run directory;
use a new name or Prime's explicit resume configuration.
