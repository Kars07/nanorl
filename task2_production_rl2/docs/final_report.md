# Final acceptance report

STATUS: **PASS**

The production capstone is operational: Prime inference/training runs inside our Modal
containers, Verifiers v1 owns the environments and traces, and untrusted repo execution is
isolated in E2B. No Prime hosted inference, training, monitoring, or sandbox API is used.
PASS means the production systems/inspection acceptance gate is implemented and executed;
it is not a claim that two tiny optimizer steps improved model capability. Compaction and
handoff failures are deterministic injections into an actual tokenized Prime sample rather
than naturally occurring policy traces, and that limitation remains explicit below.

## Pinned stack and hardware

- Prime-RL `8c1f196dd39699726ee8ff52f6ee2495c5fa38df`
- Verifiers `7251c60934d2c42af85d42a1da3da62269b7957e`
- Renderers `2846a3dcd29318c1fc98de3498bab4190997af9e`
- vLLM `0.26.0+cu129`, PyTorch `2.11.0+cu128`, CUDA image 12.8.1
- Modal A10 for inference/eval and A10:2 for trainer plus inference
- E2B microVM per repo-repair rollout, internet disabled by default
- local Docker unavailable; user-directed E2B replaces the production sandbox path

## Verifiers and environments

- `tiny-repo-repair-v1`: 150 typed deterministic instances with disjoint 100/25/25
  train/validation/test splits, generator loading, hidden commands absent from guest files,
  and an objective E2B final-state verifier. Executed smoke runs use bounded subsets.
- Terminal, browser, structured tool-use, and ordered long-horizon packages were executed
  with self-hosted Qwen3-1.7B and each earned reward 1.0. The browser success required
  search, an environment-enforced open-page step, evidence extraction, and submission;
  `pages_visited=1`.
- `proposer-solver-v1` is a real `vf.Env`: one proposer plus two concurrent traces of the
  same `solver` role. All three actual traces earned 1.0, with `episode_agents=3` and
  `solver_passes=2`.
- Built-in `null` and `bash` plus custom `learning-harness` ran through actual `uv run eval`
  commands against Prime inference. All produced valid traces; their tiny exact-answer
  policy reward was 0.
- GEPA used six real rollouts and reflection, generated three candidates, and retained the
  seed because every validation reward was 0.
- Harbor 0.20's official `harbor/hello-world` ran end-to-end: self-hosted Prime inference
  produced a typed command, E2B executed it, upstream `HarborTask.score` staged and ran the
  real verifier, and trace `843c921fb2604df7bcfa4ea69ce6a282` received reward 1.0. The
  adapter rejects Dockerfiles beyond this task's audited `FROM` + `WORKDIR` subset.

## Prime-RL results

- Official reverse-text: 20 optimizer steps; checkpoint/resume 20→21; HF export; actual
  Prime checkpoint reload and request.
- Repo repair: two optimizer steps; rewards 0.50/0.25; losses -0.0865/-0.0968; gradient
  norms 3.4033/4.3005; throughput 230/710 token/s; mismatch KL 0.0006/0.0007.
- Held-out RL checkpoint: 4/4 traces completed, zero provider errors, reward 0/4. This proves
  the training/reload path, not learned capability.
- The actual Task1 SFT checkpoint and Qwen3-1.7B reference each completed the same four
  held-out E2B tasks with zero provider errors and reward 0/4; neither submitted a repair.
- Tensor hashes across the resumed step changed for embeddings and layer-0 Q projection;
  the selected final norm did not change. Prime's in-node weight transport was NCCL.
- The four-call repo trace became a real Prime sample: 2,178 tokens, 1,555 trainable,
  623 context, four sampled spans, aligned logprobs, and nonzero GRPO advantage.
- MAX-RL completed an optimizer step on four E2B repo rollouts (rewards `[0,0,0,1]`):
  Prime assigned advantages `[-1,-1,-1,+3]`; trainer loss was -0.1042.
- Hierarchical GRPO completed an optimizer step on proposer/solver episodes. Prime's replay
  showed episode-local solver advantages (`[1,0]` became `[+0.5,-0.5]`); trainer loss was
  0.1463 and 10/18 effective traces carried nonzero credit.
- ECHO completed an optimizer step on a zero-reward terminal cohort. Prime assigned three
  saved tool-observation tokens CE weight 0.1; trainer loss was 1.2717 despite zero RL
  advantages.
- The corrected per-environment run completed two optimizer steps: terminal/ECHO produced
  loss 1.2510 from four CE tokens, then browser/GRPO produced loss 0.0000 because all four
  rewards and advantages were equal. Both environments survived filtering and each was
  100% of one recorded step.

## Configuration and diagnostics

- Real `rl --dry-run` passed for repo repair and the 3:1:1 multi-source config.
- Prime's actual `TrainSource` yielded 59.57/20.74/19.69% over 10,000 draws and restored
  its RNG sequence from `state_dict()`.
- MaxRL, hierarchical GRPO, ECHO, and per-environment GRPO/ECHO typed configs all pass
  Prime's actual dry-run and have actual optimizer evidence plus source-level sample replay.
- Five real E2B failures and seventeen artifact/source detectors pass. Compaction, handoff,
  renderer, contamination, domination, omitted update, and wrong-resume cases are deliberate
  mutations of real artifacts where natural occurrence was not available.
- The full 100-train/25-validation production config passed Prime's actual `rl @ ...
  --dry-run`; its resolved child config records those exact slices. The sealed 25-task test
  split exists only in `configs/eval/repo_repair_test.toml` at indices 125â€“149.
- Local tests: 32 passed. Owner-scoped E2B cleanup reports zero live sandboxes.

## Most important bugs found

1. SIGTERM bypassed MCP async cleanup and leaked paid E2B VMs; signal cleanup plus an
   owner-scoped recovery command fixed it.
2. Exported checkpoint paths could not infer the Qwen tool parser; explicit `hermes` and
   auto-tool-choice fixed the HTTP 400.
3. Exact browser answer matching rejected a semantically correct numeric phrase; numeric
   word/digit normalization was added and regression-tested.
4. Hierarchical GRPO imports the legacy `proposer_solver` module name even for a versioned
   taskset ID; a compatibility export of the same EnvConfig class satisfied the contract.
5. Prime's default zero-advantage filter discarded ECHO samples that still carried a CE
   observation signal; disabling that filter for ECHO/mixed experiments restored training.

## Read first

1. `modal_apps/self_hosted_rollout.py`
2. `environments/tiny_repo_repair_v1/tiny_repo_repair_v1/taskset.py`
3. `environments/tiny_repo_repair_v1/tiny_repo_repair_v1/servers/tool.py`
4. `configs/rl/repo_repair_smoke.toml`
5. `docs/multi_turn_training.md`

## Remaining limitations

- Harbor execution is proven only for the audited hello-world Dockerfile subset; arbitrary
  Dockerfile builds, separate verifier images, and broader Harbor tasksets are not yet covered.
- The browser half of the mixed algorithm run had all-equal zero rewards, so it validates
  dispatch and zero-gradient GRPO semantics but not browser-policy improvement.
- Context compaction and sub-agent handoff detectors use mutations of a real tokenized
  sample, not naturally produced training traces.
- Two repo-repair steps are a systems smoke; held-out reward and both comparison baselines
  remained zero.
- Saved trainer logs expose aggregate mismatch KL rather than per-token trainer logprobs.
