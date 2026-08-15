# Self-hosted Prime inference

The command actually run on Modal is:

```text
uv run --frozen inference --vllm.model Qwen/Qwen3-0.6B \
  --vllm.max-model-len 4096 --vllm.gpu-memory-utilization 0.85
```

Prime starts a vLLM engine on 8100 and its router on 8000. The client waits for
`http://127.0.0.1:8000/health`, then Verifiers uses
`http://127.0.0.1:8000/v1`. Actual logs show the router discovering the one worker,
consistent-hash routing by `x-session-id`, and successful chat-completions requests.

This entrypoint is preferred to bare `vllm serve` because it also exposes the RL weight
update/broadcaster endpoints. The model, torch, vLLM, router, CUDA base, and flash-attn wheel
are pinned. No endpoint URL or API credential from Prime hosted infrastructure exists in
the code.
