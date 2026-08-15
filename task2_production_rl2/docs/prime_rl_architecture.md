# Prime-RL architecture observed on Modal

The actual `uv run --frozen rl` launcher split one TOML into inference, orchestrator,
trainer, and per-environment configs. GPU 0 hosted Prime's vLLM server/router; GPU 1 hosted
the FSDP trainer; the orchestrator ran on CPU in the same Modal machine.

For each group, the orchestrator dispatched Verifiers episodes, received traces, converted
them to rollouts/training samples, computed group credit, applied filters, and sent packed
token streams over the rollout transport. The trainer recomputed logprobs, calculated the
importance ratio against sampling logprobs, applied masks/loss streams, stepped the
optimizer, and broadcast new weights. The official run logged in-flight updates through
policy v18 and max off-policy lag 2. Inference exposes `/update_weights` and
`/init_broadcaster`; this is why the documented Prime inference entrypoint is used instead
of bare `vllm serve`.

The completed official control produced `checkpoints/step_20`, `weights/step_20/STABLE`,
and a reloadable Hugging Face-format `model.safetensors` plus tokenizer/config files.
