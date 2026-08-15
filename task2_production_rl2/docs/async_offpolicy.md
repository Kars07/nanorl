# Async and off-policy behavior

The official run visibly overlapped rollout collection and optimization. While a batch was
being assembled, 128 rollouts remained in flight. Completed steps reported maximum
off-policy lag 2 and the orchestrator logged in-flight policy updates (`v5`, `v6`, …,
`v18`).

`dispatcher.py` stamps a group's policy version at dispatch. The orchestrator computes
rollout age relative to the trainer step. The trainer compares its logprob to the saved
inference logprob; `exp(trainer - inference)` is the importance ratio. Large mismatch is
both a diagnostic and a trust-mask input. `probes/inspect_policy_versions.py` extracts the
observed step/reward/lag transitions from the actual log.
