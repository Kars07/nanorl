# Prime-RL configuration

Prime-RL composes TOML with CLI overrides through typed Pydantic configs. The real dry-run
command for the custom environment is:

```text
uv run --frozen rl @ /opt/configs/repo_repair_smoke.toml \
  --output-dir /outputs/repo-repair-dry-run --no-wandb --dry-run
```

It wrote inference, trainer, orchestrator, train-environment, and held-out-environment
subconfigs. Top-level `model`, `seq_len`, `max_steps`, `ckpt`, and `env_vars` propagate to
components unless a narrower component value overrides them. Environment runtime and E2B
settings remain under each source. Train/eval source names are unique and explicit.

The E2B secret itself is never placed in TOML. Config contains only the non-secret path
`/tmp/task2-e2b-key`; the Modal wrapper creates that file with mode 0600.

## Multi-source and algorithm validation

`configs/rl/multi_source_3_1_1.toml` is a real three-source Prime config. Its
`uv run --frozen rl @ ... --dry-run` invocation exited 0 and wrote all resolved child
configs. The pinned `prime_rl.orchestrator.train_source.TrainSource` was then sampled
10,000 times, producing repo/terminal/browser frequencies 0.5957/0.2074/0.1969 against
expected 0.6/0.2/0.2; restoring `state_dict()` reproduced the next 20 draws.

The same CLI accepts `max_rl_repo_repair.toml`,
`hierarchical_grpo_proposer_solver.toml`, and `per_environment_grpo_echo.toml`, proving
typed MaxRL selection, Prime's proposer-solver validation, and a run where browser inherits
GRPO while terminal overrides it with ECHO. The hierarchical validator imports the legacy
module name `proposer_solver`; the versioned package exposes the identical EnvConfig class
through that compatibility module, so Prime's actual `isinstance` check passes without an
upstream framework patch.
