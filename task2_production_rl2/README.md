# Task 2 — production agentic RL

This repository runs the real pinned Prime-RL and Verifiers v1 entrypoints on infrastructure
owned by this project:

```text
Modal A10 GPU(s)
  └─ Prime `uv run inference` + `uv run rl`
      └─ Verifiers v1 workers and multi-turn traces
          └─ custom MCP toolset
              └─ isolated E2B microVM per rollout (internet off by default)
```

No Prime hosted inference, hosted training, or Prime sandbox is used. The workspace's
`prime_rl_api` value is never read or forwarded. Only the E2B key enters the Modal function;
it is placed in a mode-0600 host file and never exposed to the model or guest.

## Executed results

- Modal and Modal→E2B smoke tests passed.
- The official Prime reverse-text run completed 20 optimizer steps on `A10:2`; reward reached
  roughly 0.74–0.82. Checkpoint resume continued from step 20 to 21, and exported step-20
  weights reloaded into the actual Prime inference command and answered a request.
- The adapted repo-repair GRPO run completed two optimizer steps on `A10:2`, using real
  multi-turn Verifiers traces and E2B sandboxes. Training rewards were 0.50 then 0.25; losses
  were -0.0865 and -0.0968; aggregate mismatch KL was 0.0006 and 0.0007.
- A successful four-call trace became one real Prime `TrainingSample`: 2,178 tokens, 1,555
  trainable tokens, 623 context tokens, four sampled spans, and nonzero GRPO advantages.
- Exported repo-repair `step_2` weights reloaded successfully. Four held-out E2B episodes had
  zero provider errors but reward 0.00; the tiny two-step smoke run did not improve held-out
  capability.
- Verifiers v1 GEPA ran its complete six-rollout budget against self-hosted vLLM. Three
  candidates failed to beat the 0.00 baseline, so the seed prompt remained best.
- Harbor 0.20.0's official `harbor/hello-world` package ran and was graded on E2B from
  inside the Modal driver using self-hosted Prime inference. The upstream verifier gave
  reward 1.0. The adapter is limited to the audited `FROM` + `WORKDIR` Dockerfile subset
  and rejects richer Dockerfiles.
- The Task1 SFT checkpoint and Qwen3-1.7B reference were both served by Prime inference and
  evaluated on the same four held-out E2B tasks. Both completed without provider errors and
  both scored 0/4, so neither is presented as a capability win.
- The real Prime `TrainSource` produced a 59.57/20.74/19.69 percent split over 10,000 draws
  for the configured 3:1:1 repo/terminal/browser ratios, and its saved RNG state reproduced
  the next draw sequence. Prime's `rl --dry-run` accepted the corresponding full config.
- Actual Verifiers CLI runs exercised built-in `null` and `bash`, the custom learning
  harness, terminal/browser/tool/long-horizon packages, and a three-trace proposer-solver
  episode. Saved summaries list called tools separately from registered schemas.
- The failure lab passes five real E2B infrastructure/verifier cases and seventeen
  artifact-backed detectors, including mutations of the real four-call training sample.
- Prime optimizer runs also completed for MAX-RL, hierarchical GRPO, ECHO, and a two-step
  per-environment ECHO/GRPO configuration. The mixed run exposed and fixed a real filter
  bug: advantage-only filtering removed CE-bearing ECHO samples. Saved Prime replay shows
  four CE tokens at weight 0.1 in terminal step 1 and no CE/nonzero-advantage tokens in the
  all-zero browser GRPO step 2.
- The repo-repair dataset contains 150 deterministic instances with tested, disjoint
  100/25/25 train/validation/test slices. Prime's actual dry-run accepted the production
  train+validation config; the test slice is kept in a separate evaluation-only config.

## Reproduce

```powershell
uv sync
uv run pytest -q
$env:PYTHONIOENCODING='utf-8'

uv run modal run modal_apps/modal_smoke.py
uv run modal run modal_apps/e2b_smoke.py
uv run modal run modal_apps/run_verifiers.py --mode dry-run
uv run modal run modal_apps/self_hosted_rollout.py --mode train-official
uv run modal run modal_apps/self_hosted_rollout.py --mode resume-official
uv run modal run modal_apps/self_hosted_rollout.py --mode reload-official
uv run modal run modal_apps/self_hosted_rollout.py --mode dry-run-repo-repair
uv run modal run modal_apps/self_hosted_rollout.py --mode dry-run-production
uv run modal run modal_apps/self_hosted_rollout.py --mode train-repo-repair
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-sample
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-weight-fingerprints
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-repo-checkpoint
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-environment-suite
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-browser
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-harness-suite
uv run modal run modal_apps/self_hosted_rollout.py --mode eval-model-comparison
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-source-mixing
uv run modal run modal_apps/self_hosted_rollout.py --mode dry-run-multi-source
uv run modal run modal_apps/self_hosted_rollout.py --mode dry-run-algorithms
uv run modal run modal_apps/self_hosted_rollout.py --mode train-max-rl
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-max-rl-sample
uv run modal run modal_apps/self_hosted_rollout.py --mode train-hierarchical-grpo
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-hierarchical-samples
uv run modal run modal_apps/self_hosted_rollout.py --mode train-echo-terminal
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-echo-samples
uv run modal run modal_apps/self_hosted_rollout.py --mode train-per-env-grpo-echo
uv run modal run modal_apps/self_hosted_rollout.py --mode inspect-per-env-samples
uv run modal run modal_apps/self_hosted_rollout.py --mode gepa-dry-run
uv run modal run modal_apps/self_hosted_rollout.py --mode gepa
uv run modal run modal_apps/e2b_smoke.py --mode failure-lab

uv run --with harbor==0.20.0 python scripts/inspect_harbor.py
uv run modal run -q modal_apps/self_hosted_rollout.py --mode eval-harbor-e2b
uv run python probes/inspect_trace.py artifacts/rollouts/repo_repair_mixed_step1/all_traces.jsonl
uv run python probes/inspect_training_sample.py
uv run python probes/inspect_interleaving.py
uv run python probes/inspect_logprob_mismatch.py
uv run python scripts/cleanup_e2b.py       # owner-scoped dry-run
```

The large model weights remain in the named Modal volume `task2-prime-rl-outputs`; local
`artifacts/` contains logs, traces, configs, summaries, stable markers, and GEPA outputs.
Exact revisions are in `upstream/REVISIONS.md` and the honest gate report is
`docs/final_report.md`.
