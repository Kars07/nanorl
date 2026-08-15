# Failure-injection laboratory

The laboratory separates failures that require a real E2B microVM from structural
mutations applied to real Prime-RL artifacts. It does not claim that a fabricated token
mutation was observed naturally during training.

## Real E2B failures

`modal_apps/e2b_smoke.py --mode failure-lab` created disposable E2B sandboxes from a
Modal function and asserted:

- a public test can pass while an impossible hidden assertion fails;
- hidden verifier material is absent from the agent-visible filesystem;
- a required network operation times out when the sandbox has internet disabled;
- an actual E2B command timeout is surfaced and normalized to exit code 124;
- importing an absent package fails with a non-zero process exit.

The raw structured results are in `artifacts/failure_injections/e2b.json`. The network
and runtime cases use the E2B SDK's real `TimeoutException`; they are not mocked.

## Artifact-backed detectors

`probes/failure_injection_lab.py` loads the saved four-call Prime training sample, the
actual repo-repair trainer and resume logs, the actual rollout policy versions, and the
real train/eval config. It then checks real controls and deliberately injected variants:

| Case | Evidence type | Detector |
| --- | --- | --- |
| identical rewards, group size one, zero advantage | Prime-compatible independent loss math | exact zero advantage/loss assertion |
| missing harness tool | real Verifiers `validate_pairing` | MCP task rejected for a harness with `SUPPORTS_MCP=False` |
| gibberish | injected token repetition | dominant-token ratio |
| extension success | real sample prefixes | prefix equality |
| extension break / compaction / handoff / renderer | mutations of real token IDs | prefix inequality |
| stale policy | real step-1 traces | all sampled policy versions equal zero |
| omitted update | mutation of real update sequence | missing final version |
| logprob mismatch | real trainer log | positive mismatch KL (0.0006, 0.0007) |
| clean split / contamination | real config plus injected overlap | set-disjointness/overlap |
| environment domination | injected 19:1 counts | share greater than 90% |
| wrong resume | real 20→21 log plus injected step 19 request | requested/restored-step mismatch |

The result is `artifacts/failure_injections/artifact_detectors.json`; every check is also
covered by `tests/test_failure_injections.py`. The source artifact is trace
`07ce2c5365bd4927bb5f7e756b41ded8`, whose sampled spans end at tokens 1031, 1254,
1807, and 2178.

## Reproduce

```powershell
uv run pytest -q tests/test_failure_injections.py
uv run python probes/failure_injection_lab.py
$env:PYTHONIOENCODING='utf-8'
uv run modal run modal_apps/e2b_smoke.py --mode failure-lab
```
