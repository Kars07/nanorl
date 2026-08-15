# Algorithms and loss semantics

Pinned GRPO assigns `reward - group_mean`; it does not divide by group standard deviation.
Therefore a group of identical rewards has exactly zero advantage. This differs from the
Task-1 teaching implementation if that implementation used standard-deviation scaling.

The current default trainer loss is not classic clipped PPO. `default_loss_fn` computes:

```text
log_ratio = trainer_logprob - inference_logprob
ratio     = exp(log_ratio)
prob_diff = exp(trainer_logprob) - exp(inference_logprob)
```

It applies a direction-aware DPPO binary trust mask based on advantage sign, uses
`-adv_tau * advantage * ratio` on kept trainable tokens, and adds
`kl_tau * log_ratio²` on trainable tokens. Optional per-token component weights are applied
after the components. `probes/loss_reference.py` independently reproduces this scalar math
and its unit tests cover GRPO, masking, ratios, KL, and loss weights.

Pinned algorithm implementations include GRPO, ECHO, MAX-RL, RAE, hierarchical GRPO,
OPD, OPSD, and SFT distillation. Algorithm selection is per environment; it is independent
of task/harness/runtime selection.

## Actual optimizer runs

All commands below used the pinned Prime `rl @ config` entrypoint inside the project's
two-A10 Modal wrapper. They are optimizer runs, not dry-run simulations.

| Algorithm | Actual cohort | Prime trainer evidence |
|---|---|---|
| MAX-RL | repo repair, 4 rollouts, rewards `[0,0,0,1]` | loss `-0.1042`, grad norm `15.6237`, 184 token/s |
| hierarchical GRPO | proposer plus two solvers per episode | loss `0.1463`, grad norm `5.7299`, 180 token/s |
| ECHO | terminal, 4 zero-reward rollouts | loss `1.2717`, grad norm `189.3151`, 134 token/s |
| per-environment ECHO then GRPO | terminal step 1, browser step 2 | losses `1.2510`, `0.0000`; grad norms `184.4154`, `0.0025` |

The inspectors replay saved traces through the pinned algorithm classes. MAX-RL assigned
`-1` to each failed rollout and `+3` to the successful rollout. Hierarchical GRPO grouped
solvers by episode: a `[1,0]` solver pair received `[+0.5,-0.5]`, while equal-reward pairs
and proposers received zero. This is why solvers from different proposed tasks must not
share one baseline.

ECHO exposed an important filter interaction. Its action tokens retain GRPO's RL stream,
but later environment-provided tool content receives a separate CE stream. The default
`zero_advantage` post-filter only sees the RL advantages. In the first mixed attempt it
discarded 40 real zero-reward rollouts before the trainer could use ECHO's CE signal. The
production configs explicitly set `pre_batch_filters=[]` and `post_batch_filters=[]` for
this experiment. A rerun then optimized successfully: the dedicated ECHO cohort contained
three CE tokens at weight `0.1`; the mixed terminal cohort contained four. The orchestrator
still reported `Trainable 0/4` because that counter measures nonzero RL advantage, not ECHO
CE tokens; the trainer's nonzero loss and the saved `ce_weights` disambiguate it.

The mixed run did not hide environment behavior behind an aggregate. Step 1 was 100%
`terminal-echo` and step 2 was 100% `browser-grpo`. Browser rewards were all zero, so GRPO
correctly produced no nonzero advantage tokens and a zero loss. This proves routing and
filter survival, but not browser learning. Exact evidence is in
`artifacts/training/algorithm_runs/{max_rl,hierarchical_grpo,echo_terminal,per_env_grpo_echo}`;
the failed-filter cohort is retained separately as
`per_env_grpo_echo_failed_zero_advantage`.
