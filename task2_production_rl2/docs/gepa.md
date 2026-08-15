# GEPA versus gradient RL

The pinned Verifiers v1 `gepa` CLI was dry-run and then executed on Modal against the local
Prime vLLM endpoint. Config: three `learning-harness` tasks, a 2/1 train/validation split,
six-rollout budget, minibatch one, and Qwen3-0.6B for both rollout and reflection. Both
clients resolve to `http://127.0.0.1:8000/v1` with a dummy local key.

The run evaluated the seed, performed three reflection iterations, proposed three prompt
candidates, and persisted `traces.jsonl`, `candidates.json`, `run_log.json`, and
`best_system_prompt.txt` under `artifacts/gepa/`. Every validation/subsample reward was 0,
so the seed prompt remained best. This is a completed, non-improving baseline.

GEPA changes the system prompt and never updates policy weights. Prime-RL instead assigns
token-level advantages and updates model parameters. The GEPA taskset is deliberately tiny
and tool-free; its result is not presented as a repo-repair improvement.
